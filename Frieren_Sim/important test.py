from typing import Literal
import torch
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from pytorch_yagaodirac_v2.Util import _tensor_equal, _float_equal
from matplotlib_wrapper.wrapper import image_show_simple_combined_style_1, data_list_2d
from matplotlib import pyplot as plt

def show(data:data_list_2d, title:str, vmin=0, vmax=1.)
    the_axes = plt.subplot()
    image_show_simple_combined_style_1(data, title="before & init", 
                                            vmin=0,vmax=1.,the_axes=the_axes,)
    plt.show()
    the_axes.cla()
    del the_axes
    pass




# if "basic calc" and True:
#     #old_anchor_at = 0. always 0
#     new_anchor_at = 0.5
#     speed = 0.5
#     !!!speed-new_anchor_at
#     how to calc speed?
#     for:
#         if repulsive
#         if dist >0.5
#         amount/dist_sqr
#         if dist <0.5
#         amount*dist_sqr
#         also, the direction.



if "basic 1d test" and False:
    DIM = 3
    PADDING = 2
    mem_1d_a:torch.Tensor = torch.zeros(size=[DIM+PADDING*2])
    mem_1d_a[2] = 1.  
    show([mem_1d_a.tolist()], title="before & init")
        
        
    mem_1d_a_move = torch.zeros(size=[PADDING*2+1, DIM])
    mem_1d_a_move[0,:] = 0.2
    mem_1d_a_move[1,:] = 0.2
    mem_1d_a_move[2,:] = 0.2
    mem_1d_a_move[3,:] = 0.2
    mem_1d_a_move[4,:] = 0.2
    
    mem_1d_a_move[0,0] = 0.1
    mem_1d_a_move[1,0] = 0.15
    mem_1d_a_move[2,0] = 0.5
    mem_1d_a_move[3,0] = 0.15
    mem_1d_a_move[4,0] = 0.1
    show(mem_1d_a_move, title="mem_1d_a_move")
    
        
    _sum = mem_1d_a_move.sum(dim=0)
    assert _tensor_equal(_sum, [1.]*(DIM))

    temp_1d = torch.zeros_like(mem_1d_a)
    source = mem_1d_a[PADDING:DIM+PADDING]
    show([source.tolist()], title="source")
    # print(mem_1d_a)
    # print(mem_1d_a_move)
    # print(source)
    
    for offset in range(PADDING*2+1):
        _this_part = mem_1d_a_move[offset,:]
        assert source.shape == _this_part.shape
        temp_1d[offset:DIM+offset] += source * _this_part
        pass
    assert _float_equal(temp_1d.sum().item(), 1.)
    assert _tensor_equal(temp_1d.sum(), mem_1d_a.sum())
    assert _tensor_equal(temp_1d, [0.1, 0.15, 0.5, 0.15, 0.1,0,0])
    show([temp_1d.tolist()], title="after")
    del DIM,PADDING,mem_1d_a,mem_1d_a_move,_sum,offset
    pass

if "correct the padding style 1, no loss" and False:
    DIM = 3
    PADDING = 2
    mem_1d_a = torch.zeros(size=[DIM+PADDING*2])
    mem_1d_a = torch.tensor([1,2,4,8,16,32,64])
    _ori_sum = mem_1d_a.sum()
    #mem_1d_a[0:5] = torch.tensor([0.1, 0.15, 0.5, 0.15, 0.1])
    show([mem_1d_a.tolist()], title="before & init", vmin=0,vmax=100.)
    
    
    #fold left
    mem_1d_a[PADDING:PADDING*2] += mem_1d_a[0:PADDING].flip(dims=[0])#the dim!!!
    mem_1d_a[0:PADDING]=0.
    assert _tensor_equal(mem_1d_a, [0.,0,6,9,16,32,64])
    assert _tensor_equal(mem_1d_a.sum(), _ori_sum)
    
    #fold right
    mem_1d_a[DIM:DIM+PADDING] += mem_1d_a[DIM+PADDING:DIM+PADDING*2].flip(dims=[0])#the dim!!!
    mem_1d_a[DIM+PADDING:DIM+PADDING*2]=0.
    assert _tensor_equal(mem_1d_a, [0.,0,6,73,48,0,0])
    assert _tensor_equal(mem_1d_a.sum(), _ori_sum)
    show([mem_1d_a.tolist()], title="after", vmin=0,vmax=100.)
    del DIM,PADDING,mem_1d_a,_ori_sum
    pass

if "diffusion 1d test, simple case" and False:
    DIM = 5
    PADDING = 1
    mem_1d_a:torch.Tensor = torch.zeros(size=[DIM+PADDING*2])
    mem_1d_a[3] = 1.  
    show([mem_1d_a.tolist()], title="before & init")

    
    #the diffusion kernal
    mem_1d_a_move = torch.zeros(size=[PADDING*2+1, DIM])
    mem_1d_a_move[0,:] = 0.1
    mem_1d_a_move[1,:] = 0.8
    mem_1d_a_move[2,:] = 0.1
        
    _sum = mem_1d_a_move.sum(dim=0)
    assert _tensor_equal(_sum, [1.]*(DIM))
    show(mem_1d_a_move, title="mem_1d_a_move")


    for _iter_count in range(3):
        temp_1d = torch.zeros_like(mem_1d_a)
        source = mem_1d_a[PADDING:DIM+PADDING]
        show([source.tolist()], title="source")

        
        for offset in range(PADDING*2+1):
            _this_part = mem_1d_a_move[offset,:]
            assert source.shape == _this_part.shape
            temp_1d[offset:DIM+offset] += source * _this_part
            pass
        assert _float_equal(temp_1d.sum().item(), 1.)
        assert _tensor_equal(temp_1d.sum(), mem_1d_a.sum())
        show([temp_1d.tolist()], title="after")

        
        #tail
        mem_1d_a = temp_1d
        pass
    
    del DIM,PADDING,mem_1d_a,mem_1d_a_move,_sum,offset
    pass

# for _ in range(50):
#     sum = torch.tensor(0.)
#     for _ in range(500):
#         the_rand = (torch.rand(size=[])-0.5)
#         sum += the_rand
#         pass
#     prin(sum)
#     pass

if "diffusion 1d test" and True:
    DIM = 15
    PADDING = 1
    mem_1d_a:torch.Tensor = torch.zeros(size=[DIM+PADDING*2])
    mem_1d_a[PADDING+DIM//2] = 1.  
    show([mem_1d_a.tolist()], title="before & init")

    
    #for _iter_count in range(3):
    while True:
        the_rand = torch.rand(size=[])*0.5+0.25
        the_rand *= torch.randint(low=0, high=2,size=[])*2-1#sign
        assert  _float_equal(the_rand,  0.5, epsilon=0.251) or\
                _float_equal(the_rand, -0.5, epsilon=0.251)
        
        #the diffusion kernal
        mem_1d_a_move = torch.zeros(size=[PADDING*2+1, DIM])
        mem_1d_a_move[0,:] = torch.max(-the_rand, torch.tensor(0.))
        mem_1d_a_move[1,:] = 1-the_rand.abs()
        mem_1d_a_move[2,:] = torch.max(the_rand, torch.tensor(0.))
            
        _sum = mem_1d_a_move.sum(dim=0)
        assert _tensor_equal(_sum, [1.]*(DIM))
        show(mem_1d_a_move, title="mem_1d_a_move")


        temp_1d = torch.zeros_like(mem_1d_a)
        source = mem_1d_a[PADDING:DIM+PADDING]
        show([source.tolist()], title="source")
        # print(mem_1d_a)
        # print(mem_1d_a_move)
        # print(source)

        
        for offset in range(PADDING*2+1):
            _this_part = mem_1d_a_move[offset,:]
            assert source.shape == _this_part.shape
            temp_1d[offset:DIM+offset] += source * _this_part
            pass
        assert _float_equal(temp_1d.sum().item(), 1.)#the total amount should keep
        assert _tensor_equal(temp_1d.sum(), mem_1d_a.sum())#the amount should keep before and after
        show([temp_1d.tolist()], title="after")

        
        #tail
        mem_1d_a = temp_1d
        
        pass
    del DIM,PADDING,mem_1d_a,mem_1d_a_move,_sum,offset
    pass



if "anti diffusion(gravity back) hyper param sweap" and True:
    DIM = 25
    PADDING = 1
    mem_1d_a:torch.Tensor = torch.zeros(size=[DIM+PADDING*2])
    mem_1d_a[PADDING+DIM//2] = 1.#in the mid.
    show([mem_1d_a.tolist()], title="before & init")
    
    
    #for _iter_count in range(3):
    while True:
        if "diffuses away":
            #new anchor
            new_anchor_at = torch.rand(size=[])*0.5+0.25
            new_anchor_at *= torch.randint(low=0, high=2,size=[])*2-1#sign
            assert  _float_equal(new_anchor_at,  0.5, epsilon=0.251) or\
                    _float_equal(new_anchor_at, -0.5, epsilon=0.251)
            
            #the kernal
            #kernal----diffusion away
            mem_1d_a_move = torch.zeros(size=[PADDING*2+1, DIM])
            mem_1d_a_move[0,:] += torch.max(new_anchor_at, torch.tensor(0.))
            mem_1d_a_move[1,:] += 1-new_anchor_at.abs()
            mem_1d_a_move[2,:] += torch.max(-new_anchor_at, torch.tensor(0.))
            #kernal----gravity back
            torch.nn.Conv1d()
            mem_1d_a[] 1w
            assert False
            
            
            
            
            _sum = mem_1d_a_move.sum(dim=0)
            assert _tensor_equal(_sum, [1.]*(DIM))
            #show(mem_1d_a_move, title="mem_1d_a_move")
            
            #buffer
            temp_1d = torch.zeros_like(mem_1d_a)
            
            #chunk the origin
            source = mem_1d_a[PADDING:DIM+PADDING]
            #show([source.tolist()], title="source")
            
            #real calc
            for offset in range(PADDING*2+1):
                _this_part = mem_1d_a_move[offset,:]
                assert source.shape == _this_part.shape
                temp_1d[offset:DIM+offset] += source * _this_part
                pass
            
            assert _float_equal(temp_1d.sum().item(), 1.)#the total amount should keep
            assert _tensor_equal(temp_1d.sum(), mem_1d_a.sum())#the amount should keep before and after
            #show([temp_1d.tolist()], title="after")
            
            #buffer back to mem
            mem_1d_a = temp_1d
            pass
        show([mem_1d_a.tolist()], title="after diffusion")
        
        
        
        
        
        
        
        
        if "protect back":
            _ori_sum = mem_1d_a.sum()
            #show([mem_1d_a.tolist()], title="before & init", vmin=0,vmax=100.)
            
            #fold left
            mem_1d_a[PADDING:PADDING*2] += mem_1d_a[0:PADDING].flip(dims=[0])#the dim!!!
            mem_1d_a[0:PADDING]=0.
            assert _tensor_equal(mem_1d_a.sum(), _ori_sum)
            
            #fold right
            mem_1d_a[DIM:DIM+PADDING] += mem_1d_a[DIM+PADDING:DIM+PADDING*2].flip(dims=[0])#the dim!!!
            mem_1d_a[DIM+PADDING:DIM+PADDING*2]=0.
            assert _tensor_equal(mem_1d_a.sum(), _ori_sum)
            #show([mem_1d_a.tolist()], title="after")
            del _ori_sum
            pass
        show([mem_1d_a.tolist()], title="after protection")
        
        pass
    del DIM,PADDING,mem_1d_a,mem_1d_a_move,_sum,offset
    pass


















    
type correct_style_type=Literal["no loss & mirror back"]
type mem_state_type=Literal["to update", "to correct"]#finite state machine.
class grid_mem_1d():
    '''Basically, this 1d version is useless. It's designed for test.'''
    data:torch.Tensor
    correct_mode:correct_style_type
    state:mem_state_type
    def __init__(self, dim:int, padding:int, correct_mode:correct_style_type):
        assert dim > padding*4
        self.dim = dim
        self.padding = padding
        self.total_dim = dim+padding*2
        self.data = torch.zeros(size=[self.total_dim])        
        self.correct_mode = correct_mode
        self.state = "to update"
        pass
    def tick(self, tich_count = 1)->None:
        assert False
        pass
        
        
        
        
        
        
        # DIM = self.dim  
        # PADDING = self.padding
        # mem_1d_a:torch.Tensor = torch.zeros(size=[DIM+PADDING*2])
        # mem_1d_a[2] = 1.  
            
        # _sum = mem_1d_a_move.sum(dim=0)
        # assert _tensor_equal(_sum, [1.]*(DIM))

        # temp_1d = torch.zeros_like(mem_1d_a)
        # source = mem_1d_a[PADDING:DIM+PADDING]
        # if "
        
        # for offset in range(PADDING*2+1):
        #     _this_part = mem_1d_a_move[offset,:]
        #     assert source.shape == _this_part.shape
        #     temp_1d[offset:DIM+offset] += source * _this_part
        #     pass
        # assert _float_equal(temp_1d.sum().item(), 1.)
        # assert _tensor_equal(temp_1d.sum(), mem_1d_a.sum())
        # assert _tensor_equal(temp_1d, [0.1, 0.15, 0.5, 0.15, 0.1,0,0])

        # mem_1d_a = torch.zeros(size=[DIM+PADDING*2])
        # mem_1d_a = torch.tensor([1,2,4,8,16,32,64])
        # _ori_sum = mem_1d_a.sum()
        
        # #fold left
        # mem_1d_a[PADDING:PADDING*2] += mem_1d_a[0:PADDING].flip(dims=[0])#the dim!!!
        # mem_1d_a[0:PADDING]=0.
        # assert _tensor_equal(mem_1d_a, [0.,0,6,9,16,32,64])
        # assert _tensor_equal(mem_1d_a.sum(), _ori_sum)
        
        # #fold right
        # mem_1d_a[DIM:DIM+PADDING] += mem_1d_a[DIM+PADDING:DIM+PADDING*2].flip(dims=[0])#the dim!!!
        # mem_1d_a[DIM+PADDING:DIM+PADDING*2]=0.
        # assert _tensor_equal(mem_1d_a, [0.,0,6,73,48,0,0])
        # assert _tensor_equal(mem_1d_a.sum(), _ori_sum)













# mem_a = torch.zeros(size=[5,5])
# mem_a[0,0] = 1.

# mem_a_move = torch.zeros(size=[mem_a.shape, mem_a.shape, 2, 2])
# mem_a_move[:,:,0,0] = 0.9
# mem_a_move[:,:,1,0] = 0.1





# for _ in range(10):
#     _temp = torch.zeros_like(a)
#     _temp[1:,1:] = a[:1,:1]*0.25
#     _temp[1:,1:] = a[:1,:1]*0.25









# a = torch.zeros(size=[29,29])
# a[13:16,13:16] = 1.


# for _ in range(10):
    # _temp = torch.zeros_like(a)
    # _temp[1:,1:] = a[:1,:1]*0.25
    # _temp[1:,1:] = a[:1,:1]*0.25