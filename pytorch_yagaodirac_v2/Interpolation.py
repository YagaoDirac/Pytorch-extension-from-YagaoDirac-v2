import torch
import random
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from pytorch_yagaodirac_v2.Util import str_the_list
#from pytorch_yagaodirac_v2.timeit_yagaodirac import timeit

def __DEBUG_ME__()->bool:
    return __name__ == "__main__"
# if "test" and True:
#     assert __DEBUG_ME__()
#     pass

import sys
def _line_():
    caller_s_frame = sys._getframe(1)
    caller_s_line_number = caller_s_frame.f_lineno
    assert caller_s_line_number is not None
    return caller_s_line_number#######
if "test" and False:
    a = _line_()
    b = _line_()
    c = _line_()
    pass







"linear interpolation"

#def interpolation(a:torch.Tensor, b:torch.Tensor, t:torch.Tensor)->torch.Tensor:
def interpolation(a:torch.Tensor, b:torch.Tensor, t:torch.Tensor)->torch.Tensor:
    return a*(1.-t)+b*t
if "test" and __DEBUG_ME__() and False:
    def ____test____interpolation():
        a = torch.tensor([0.,4])
        b = torch.tensor([16.,16])
        t = torch.tensor([0.])
        result = interpolation(a,b,t)
        assert result.eq(a).all()
        
        aa = torch.tensor([0.,4])
        b = torch.tensor([16.,16])
        t = torch.tensor([1.])
        result = interpolation(a,b,t)
        assert result.eq(b).all()
        
        a = torch.tensor([0.,4])
        b = torch.tensor([16.,16])
        t = torch.tensor([0.5])
        result = interpolation(a,b,t)
        assert result.eq(torch.tensor([8.,10])).all()
        
        a = torch.tensor([0.,4])
        b = torch.tensor([16.,16])
        t = torch.tensor([0.25])
        result = interpolation(a,b,t)
        assert result.eq(torch.tensor([4.,7])).all()
        
        return 
        
    ____test____interpolation()
    pass

def interpolation_of_list(the_list:torch.Tensor, the_index:torch.Tensor)->torch.Tensor:
    if the_index == the_list.shape[0] -1:
        return the_list[-1]
    
    index_floor = the_index.floor().to(torch.int64)
    index_fraction = the_index- index_floor
    anchor_1 = the_list[index_floor]
    anchor_2 = the_list[index_floor +1]
    return interpolation(anchor_1, anchor_2, index_fraction)
if "test" and __DEBUG_ME__() and False:
    def ____test____interpolation_of_list():
        the_list = torch.tensor([12.,16,20,30])
        result = interpolation_of_list(the_list, torch.tensor(0.))
        assert result.eq(12)
        result = interpolation_of_list(the_list, torch.tensor(1.))
        assert result.eq(16)
        result = interpolation_of_list(the_list, torch.tensor(2.))
        assert result.eq(20)
        result = interpolation_of_list(the_list, torch.tensor(3.))
        assert result.eq(30)
        result = interpolation_of_list(the_list, torch.tensor(0.25))
        assert result.eq(13)
        result = interpolation_of_list(the_list, torch.tensor(0.5))
        assert result.eq(14)
        result = interpolation_of_list(the_list, torch.tensor(0.25))
        assert result.eq(13)
        result = interpolation_of_list(the_list, torch.tensor(1.25))
        assert result.eq(17)
        result = interpolation_of_list(the_list, torch.tensor(2.25))
        assert result.eq(22.5)
        
        return 
    
    ____test____interpolation_of_list()
    pass

def interpolation_of_list_2d(the_list:torch.Tensor, row_index:torch.Tensor, col_index:torch.Tensor)->torch.Tensor:
    assert the_list.shape.__len__() == 2
    index_floor_row = row_index.floor().to(torch.int64)
    index_floor_col = col_index.floor().to(torch.int64)
    small_chunk = the_list[index_floor_row:index_floor_row+2, index_floor_col:index_floor_col+2]
    
    index_fraction_row = row_index - index_floor_row
    index_fraction_col = col_index - index_floor_col
    if small_chunk.shape[0] == 2:
        halfway = small_chunk[0]*(1.-index_fraction_row) + small_chunk[1]*index_fraction_row
        pass
    else:
        halfway = small_chunk.reshape([-1])
        pass
    
    if halfway.shape[0] == 2:
        result = halfway[0]*(1.-index_fraction_col) + halfway[1]*index_fraction_col
        pass
    else:
        result = halfway.reshape([])
        pass
        
    return result
if "test" and __DEBUG_ME__() and False:
    def ____test____interpolation_of_list_2d():
        the_list = torch.tensor([[    10.,  20,  70],
                                    [110., 120, 170],
                                    [310., 320, 370],])
        result = interpolation_of_list_2d(the_list, torch.tensor(0.), torch.tensor(0.), )
        assert result.eq(10)
        result = interpolation_of_list_2d(the_list, torch.tensor(1.), torch.tensor(0.), )
        assert result.eq(110)
        result = interpolation_of_list_2d(the_list, torch.tensor(0.), torch.tensor(1.), )
        assert result.eq(20)
        result = interpolation_of_list_2d(the_list, torch.tensor(1.), torch.tensor(1.), )
        assert result.eq(120)
        result = interpolation_of_list_2d(the_list, torch.tensor(0.25), torch.tensor(0.), )
        assert result.eq(35)
        result = interpolation_of_list_2d(the_list, torch.tensor(0.5), torch.tensor(0.), )
        assert result.eq(60)
        result = interpolation_of_list_2d(the_list, torch.tensor(0.), torch.tensor(0.25), )
        assert result.eq(12.5)
        result = interpolation_of_list_2d(the_list, torch.tensor(0.), torch.tensor(0.5), )
        assert result.eq(15)
        
        result = interpolation_of_list_2d(the_list, torch.tensor(0.25), torch.tensor(0.25), )
        assert result.shape.__len__() == 0
        assert result.eq(37.5)
        
        result = interpolation_of_list_2d(the_list, torch.tensor(2.), torch.tensor(0.), )
        assert result.shape.__len__() == 0
        assert result.eq(310)
        result = interpolation_of_list_2d(the_list, torch.tensor(0.), torch.tensor(2.), )
        assert result.shape.__len__() == 0
        assert result.eq(70)
        return 
    
    ____test____interpolation_of_list_2d()
    pass

def reverse_interpolation(a:torch.Tensor, b:torch.Tensor, input:torch.Tensor, clamp = False)->torch.Tensor:
    result = (input-a)/(b-a)
    if clamp:
        result.clamp_(0., 1.)
        pass
    return result
if "test" and __DEBUG_ME__() and False:
    def ____test____reverse_interpolation():
        a = torch.tensor([0., 4])
        b = torch.tensor([16.,16])
        input = torch.tensor([0., 4.])
        result = reverse_interpolation(a,b,input)
        assert result.eq(0.).all()
        
        a = torch.tensor([0., 4])
        b = torch.tensor([16.,16])
        input = torch.tensor([16.,16])
        result = reverse_interpolation(a,b,input)
        assert result.eq(1.).all()
        
        a = torch.tensor([0., 4])
        b = torch.tensor([16.,16])
        input = torch.tensor([8.,10])
        result = reverse_interpolation(a,b,input)
        assert result.eq(0.5).all()
        
        a = torch.tensor([0., 4])
        b = torch.tensor([16.,16])
        input = torch.tensor([12.,13])
        result = reverse_interpolation(a,b,input)
        assert result.eq(0.75).all()
        
        a = torch.tensor([0., 4])
        b = torch.tensor([16.,16])
        input = torch.tensor([8.,16])
        result = reverse_interpolation(a,b,input)
        assert result.eq(torch.tensor([0.5, 1.])).all()
        
        
        a = torch.tensor([0.])
        b = torch.tensor([1.])
        input = torch.tensor([-0.5])
        result = reverse_interpolation(a,b,input)
        assert result.eq(-0.5)
        
        a = torch.tensor([0.])
        b = torch.tensor([1.])
        input = torch.tensor([-0.5])
        result = reverse_interpolation(a,b,input, True)
        assert result.eq(0.)
        
        return 
    
    ____test____reverse_interpolation()
    pass

def _reverse_interpolation_of_list__sequencial__list_must_sorted(the_list:torch.Tensor, list_is_Ascending:bool, the_input:torch.Tensor, 
                                Im_sure_the_list_is_sorted = False)->torch.Tensor:
    '''the list must be sorted!!!'''
    if not Im_sure_the_list_is_sorted:
        if list_is_Ascending:
            assert the_list[:-1].le(the_list[1:]).all()
            pass
        else:#decending
            assert the_list[:-1].ge(the_list[1:]).all()
            pass
        pass
    
    if list_is_Ascending:
        for ii in range(the_list.shape[0]-1):
            _right_element = the_list[ii+1]
            if the_input<_right_element:
                _left_element = the_list[ii]
                result = reverse_interpolation(_left_element, _right_element, the_input, clamp=True)
                return ii+result
            #no tail
            pass#for ii
        #not found
        return the_list.shape[0]-1
    else:#decending
        for ii in range(the_list.shape[0]-1):
            _right_element = the_list[ii+1]
            if the_input>_right_element:
                _left_element = the_list[ii]
                result = reverse_interpolation(_left_element, _right_element, the_input, clamp=True)
                return ii+result
            #no tail
            pass#for ii
        #not found
        return the_list.shape[0]-1
    pass# if the_list.shape[0]<=4

if "test" and __DEBUG_ME__() and False:
    def ____test____reverse_interpolation_of_list__sequencial__list_must_sorted():
        the_list = torch.tensor([12.,16,20,30])
        result = _reverse_interpolation_of_list__sequencial__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(12.))
        assert _tensor_equal(result, [0.])
        result = _reverse_interpolation_of_list__sequencial__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(11.))
        assert _tensor_equal(result, [0.])
        result = _reverse_interpolation_of_list__sequencial__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(30.))
        assert _tensor_equal(result, [3.])
        result = _reverse_interpolation_of_list__sequencial__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(31.))
        assert _tensor_equal(result, [3.])
        
        result = _reverse_interpolation_of_list__sequencial__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(14.))
        assert _tensor_equal(result, [0.5])
        result = _reverse_interpolation_of_list__sequencial__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(13.))
        assert _tensor_equal(result, [0.25])
        result = _reverse_interpolation_of_list__sequencial__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(17.))
        assert _tensor_equal(result, [1.25])
        result = _reverse_interpolation_of_list__sequencial__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(28.))
        assert _tensor_equal(result, [2.8])
        
        
        the_list = torch.tensor([12.,16,20,30])*-1.
        result = _reverse_interpolation_of_list__sequencial__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(-12.))
        assert _tensor_equal(result, [0.])
        result = _reverse_interpolation_of_list__sequencial__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(-11.))
        assert _tensor_equal(result, [0.])
        result = _reverse_interpolation_of_list__sequencial__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(-30.))
        assert _tensor_equal(result, [3.])
        result = _reverse_interpolation_of_list__sequencial__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(-31.))
        assert _tensor_equal(result, [3.])
        
        result = _reverse_interpolation_of_list__sequencial__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(-14.))
        assert _tensor_equal(result, [0.5])
        result = _reverse_interpolation_of_list__sequencial__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(-13.))
        assert _tensor_equal(result, [0.25])
        result = _reverse_interpolation_of_list__sequencial__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(-17.))
        assert _tensor_equal(result, [1.25])
        result = _reverse_interpolation_of_list__sequencial__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(-28.))
        assert _tensor_equal(result, [2.8])
        
        return 
    
    ____test____reverse_interpolation_of_list__sequencial__list_must_sorted()
    pass

def _reverse_interpolation_of_list__binary_search__list_must_sorted(the_list:torch.Tensor, list_is_Ascending:bool, the_input:torch.Tensor, 
                                Im_sure_the_list_is_sorted = False)->torch.Tensor:
    '''the list must be sorted!!!'''
    if not Im_sure_the_list_is_sorted:
        if list_is_Ascending:
            assert the_list[:-1].le(the_list[1:]).all()
            pass
        else:#decending
            assert the_list[:-1].ge(the_list[1:]).all()
            pass
        pass
    
    n_element = the_list.shape[0]

    if list_is_Ascending:
        if the_input.le(the_list[0]):
            return torch.tensor(0., device=the_list.device)
        if the_input.ge(the_list[-1]):
            return torch.tensor(n_element-1., device=the_list.device)
        
        _left_index:int = 0
        _right_index:int = n_element-1
        while _left_index<_right_index-1:
            _mid_index:int = (_left_index+_right_index)//2
            mid_element = the_list[_mid_index]
            
            if mid_element<the_input:
                _left_index = _mid_index
                #old code
                # if _mid_index == _right_index -1:
                #     _left_index = _mid_index
                #     break
                # _left_index = _mid_index+1
                continue
            
            else:# the_input<=mid_element:
                _right_index = _mid_index
                #old code
                # if _left_index +1 == _mid_index :
                #     _right_index = _mid_index
                #     break
                # _right_index = _mid_index-1
                continue
            # else:#equal
            #     return torch.tensor(_mid_index, dtype=torch.float32, device=the_list.device)
            pass#while
        
        assert _left_index == _right_index-1 # they are int
        fraction = reverse_interpolation(the_list[_left_index], the_list[_right_index], the_input, clamp=True)
        return fraction + _left_index
    else:# list_is_  decending
        if the_input.ge(the_list[0]):
            return torch.tensor(0., device=the_list.device)
        if the_input.le(the_list[-1]):
            return torch.tensor(n_element-1., device=the_list.device)
        
        _left_index:int = 0
        _right_index:int = n_element-1
        while _left_index<_right_index-1:
            _mid_index:int = (_left_index+_right_index)//2
            mid_element = the_list[_mid_index]
            
            if the_input<mid_element:
                _left_index = _mid_index
                continue
            else:# the_input>=mid_element
                _right_index = _mid_index
                continue
            pass#while
        
        assert _left_index == _right_index-1 # they are int
        fraction = reverse_interpolation(the_list[_left_index], the_list[_right_index], the_input, clamp=True)
        return fraction + _left_index
        pass
    pass

if "test" and __DEBUG_ME__() and False:
    def ____test____reverse_interpolation_of_list__binary_search__list_must_sorted():
        
        if "ascending" and True:
            #long list
            the_list = torch.tensor([12.,16,20,30])
            result = _reverse_interpolation_of_list__binary_search__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(12.))
            assert _tensor_equal(result, [0.])
            result = _reverse_interpolation_of_list__binary_search__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(11.))
            assert _tensor_equal(result, [0.])
            result = _reverse_interpolation_of_list__binary_search__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(30.))
            assert _tensor_equal(result, [3.])
            result = _reverse_interpolation_of_list__binary_search__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(31.))
            assert _tensor_equal(result, [3.])
            
            result = _reverse_interpolation_of_list__binary_search__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(14.))
            assert _tensor_equal(result, [0.5])
            result = _reverse_interpolation_of_list__binary_search__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(13.))
            assert _tensor_equal(result, [0.25])
            result = _reverse_interpolation_of_list__binary_search__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(17.))
            assert _tensor_equal(result, [1.25])
            result = _reverse_interpolation_of_list__binary_search__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(28.))
            assert _tensor_equal(result, [2.8])
            
            the_list = torch.arange(1000, dtype=torch.float32)
            assert the_list.nelement() == 1000
            result = _reverse_interpolation_of_list__binary_search__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(0.3))
            assert _tensor_equal(result, [0.3])
            result = _reverse_interpolation_of_list__binary_search__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(1.3))
            assert _tensor_equal(result, [1.3])
            for ii in range(999):
                result = _reverse_interpolation_of_list__binary_search__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(ii+0.))
                assert _tensor_equal(result, [ii+0.])
                pass
            result = _reverse_interpolation_of_list__binary_search__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(998.))
            assert _tensor_equal(result, [998.])
            result = _reverse_interpolation_of_list__binary_search__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(999.))
            assert _tensor_equal(result, [999.])
            result = _reverse_interpolation_of_list__binary_search__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(999.5))
            assert _tensor_equal(result, [999.])
            
            the_list = torch.arange(1000, dtype=torch.float32)+2.
            assert the_list.nelement() == 1000
            result = _reverse_interpolation_of_list__binary_search__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(2.3))
            assert _tensor_equal(result, [0.3])
            result = _reverse_interpolation_of_list__binary_search__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(3.3))
            assert _tensor_equal(result, [1.3])
            for ii in range(999):
                result = _reverse_interpolation_of_list__binary_search__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(ii+2.))
                assert _tensor_equal(result, [ii+0.])
                pass
            result = _reverse_interpolation_of_list__binary_search__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(1000.))
            assert _tensor_equal(result, [998.])
            result = _reverse_interpolation_of_list__binary_search__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(1001.))
            assert _tensor_equal(result, [999.])
            result = _reverse_interpolation_of_list__binary_search__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(1001.5))
            assert _tensor_equal(result, [999.])
            
            pass
        
        if "decending" and True:
            #long list
            the_list = torch.tensor([12.,16,20,30])*-1.
            result = _reverse_interpolation_of_list__binary_search__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(-12.))
            assert _tensor_equal(result, [0.])
            result = _reverse_interpolation_of_list__binary_search__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(-11.))
            assert _tensor_equal(result, [0.])
            result = _reverse_interpolation_of_list__binary_search__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(-30.))
            assert _tensor_equal(result, [3.])
            result = _reverse_interpolation_of_list__binary_search__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(-31.))
            assert _tensor_equal(result, [3.])
            
            result = _reverse_interpolation_of_list__binary_search__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(-14.))
            assert _tensor_equal(result, [0.5])
            result = _reverse_interpolation_of_list__binary_search__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(-13.))
            assert _tensor_equal(result, [0.25])
            result = _reverse_interpolation_of_list__binary_search__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(-17.))
            assert _tensor_equal(result, [1.25])
            result = _reverse_interpolation_of_list__binary_search__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(-28.))
            assert _tensor_equal(result, [2.8])
            
            # around 256?
            the_list = torch.arange(1000, dtype=torch.float32)*-1.
            assert the_list.nelement() == 1000
            result = _reverse_interpolation_of_list__binary_search__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(-0.3))
            assert _tensor_equal(result, [0.3])
            result = _reverse_interpolation_of_list__binary_search__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(-1.3))
            assert _tensor_equal(result, [1.3])
            for ii in range(999):
                result = _reverse_interpolation_of_list__binary_search__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(-1.*ii))
                assert _tensor_equal(result, [ii+0.])
                pass
            result = _reverse_interpolation_of_list__binary_search__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(-998.))
            assert _tensor_equal(result, [998.])
            result = _reverse_interpolation_of_list__binary_search__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(-999.))
            assert _tensor_equal(result, [999.])
            result = _reverse_interpolation_of_list__binary_search__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(-999.5))
            assert _tensor_equal(result, [999.])
            
            the_list = torch.arange(1000, dtype=torch.float32)*-1.-2.
            assert the_list.nelement() == 1000
            result = _reverse_interpolation_of_list__binary_search__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(-2.3))
            assert _tensor_equal(result, [0.3])
            result = _reverse_interpolation_of_list__binary_search__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(-3.3))
            assert _tensor_equal(result, [1.3])
            for ii in range(999):
                result = _reverse_interpolation_of_list__binary_search__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(-1.*ii-2.))
                assert _tensor_equal(result, [ii+0.])
                pass
            result = _reverse_interpolation_of_list__binary_search__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(-1000.))
            assert _tensor_equal(result, [998.])
            result = _reverse_interpolation_of_list__binary_search__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(-1001.))
            assert _tensor_equal(result, [999.])
            result = _reverse_interpolation_of_list__binary_search__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(-1001.5))
            assert _tensor_equal(result, [999.])
            
            pass
        
        return 
    
    ____test____reverse_interpolation_of_list__binary_search__list_must_sorted()
    pass


def ____old____reverse_interpolation_of_list__parallel__list_must_sorted(the_list:torch.Tensor, list_is_Ascending:bool, the_input:torch.Tensor, 
                                Im_sure_the_list_is_sorted = False)->torch.Tensor:
    '''the list must be sorted!!!
    
    This version is designed for gpu. But the cpu version is not finished. So use this for all cases at the moment.'''
    if not Im_sure_the_list_is_sorted:
        if list_is_Ascending:
            assert the_list[:-1].le(the_list[1:]).all()
            pass
        else:#decending
            assert the_list[:-1].ge(the_list[1:]).all()
            pass
        pass
    
    n_element = the_list.shape[0]
    
    # if the_list.device.type == 'cpu':
        
    #     if n_element<=4:# 4 is not well tested. It's only about performance. In cpp or rust, you may want 16?
        #     call the sequencial version.
    #     else:#too many elements. It's a binary search
            #call ____unfinished____reverse_interpolation_of_list__binary_search
    #         pass# else of n_element<=4:
    # elif the_list.device.type == 'cuda':
    
    # the above is the cpu version. it's a bit too fancy, I'll come back later.
    
    #if you have more than 16777216 element, modify the code.............. and tell me on x.com. my username is yagaodirac.
    
    #assume the list is decending. Correctness later.
    
    start_offset = 0
    _1_lshift_16 = 1<<16
    if n_element>_1_lshift_16:
        _temp_list = the_list[_1_lshift_16::_1_lshift_16]
        _temp__flag__before_sum__dim = _temp_list.lt(the_input)
        amount__list_gt = _temp__flag__before_sum__dim.sum()
        if not list_is_Ascending:
            amount__list_gt = _temp_list.nelement() - amount__list_gt
            pass
        start_offset += amount__list_gt * _1_lshift_16
        pass
    _1_lshift_8 = 1<<8
    if n_element>_1_lshift_8:
        _temp_list = the_list[start_offset+_1_lshift_8 : start_offset+_1_lshift_16 : _1_lshift_8]
        _temp__flag__before_sum__dim = _temp_list.lt(the_input)
        amount__list_gt = _temp__flag__before_sum__dim.sum()
        if not list_is_Ascending:
            amount__list_gt = _temp_list.nelement() - amount__list_gt
            pass
        start_offset += amount__list_gt * _1_lshift_8
        pass
    
    #old code
    #_temp__flag__before_sum__dim = the_list[start_offset:start_offset+_1_lshift_8].lt(the_input)
    
    _temp_list = the_list[start_offset:start_offset+_1_lshift_8]
    _temp__flag__before_sum__dim = _temp_list.lt(the_input)
    amount__list_gt = _temp__flag__before_sum__dim.sum()
    if not list_is_Ascending:
        amount__list_gt = _temp_list.nelement() - amount__list_gt 
        pass
    start_offset += amount__list_gt
    del amount__list_gt# just in case
    # if start_offset == 0:
    #     return torch.tensor(n_element-1., device=the_list.device)
    
    if start_offset == 0:
        return torch.tensor(0., device=the_list.device)
    
    if start_offset == n_element:
        return torch.tensor(n_element-1., device=the_list.device)
    fraction = reverse_interpolation(the_list[start_offset-1], the_list[start_offset], the_input, clamp=True)
    return fraction + start_offset -1
    
    # else:
    #     assert False, "unknown device????? You need to implement it yourself, or simply choose one of the previous implementations."
    
    pass# end of function

def __reverse_interpolation___perf_test_without_level(the_list:torch.Tensor, list_is_Ascending:bool, the_input:torch.Tensor, 
                                Im_sure_the_list_is_sorted = False, )->torch.Tensor:
    '''the list must be sorted!!!
    
    This version is designed for gpu. But the cpu version is not finished. So use this for all cases at the moment.'''
    if not Im_sure_the_list_is_sorted:
        if list_is_Ascending:
            assert the_list[:-1].le(the_list[1:]).all()
            pass
        else:#decending
            assert the_list[:-1].ge(the_list[1:]).all()
            pass
        pass
    
    n_element = the_list.shape[0]
    
    _temp_list = the_list
    _temp__flag__before_sum__dim = _temp_list.lt(the_input)
    amount__list_gt = _temp__flag__before_sum__dim.sum()
    if not list_is_Ascending:
        amount__list_gt = _temp_list.nelement() - amount__list_gt 
        pass
    
    if amount__list_gt == 0:
        return torch.tensor(0., device=the_list.device)
    if amount__list_gt == n_element:
        return torch.tensor(n_element-1., device=the_list.device)
    
    fraction = reverse_interpolation(the_list[amount__list_gt-1], the_list[amount__list_gt], the_input, clamp=True)
    return fraction + amount__list_gt -1

def __reverse_interpolation___perf_test_level_1(the_list:torch.Tensor, list_is_Ascending:bool, the_input:torch.Tensor, 
                                            level:int, 
                                Im_sure_the_list_is_sorted = False, )->torch.Tensor:
    '''the list must be sorted!!!
    
    This version is designed for gpu. But the cpu version is not finished. So use this for all cases at the moment.'''
    if not Im_sure_the_list_is_sorted:
        if list_is_Ascending:
            assert the_list[:-1].le(the_list[1:]).all()
            pass
        else:#decending
            assert the_list[:-1].ge(the_list[1:]).all()
            pass
        pass
    
    n_element = the_list.shape[0]
    
    start_offset = 0
    _1_lshift_16 = 1<<level
    if n_element>_1_lshift_16:
        _temp_list = the_list[_1_lshift_16::_1_lshift_16]
        _temp__flag__before_sum__dim = _temp_list.lt(the_input)
        amount__list_gt = _temp__flag__before_sum__dim.sum()
        if not list_is_Ascending:
            amount__list_gt = _temp_list.nelement() - amount__list_gt
            pass
        start_offset += amount__list_gt * _1_lshift_16
        pass
    
    _temp_list = the_list[start_offset:start_offset+_1_lshift_16]
    _temp__flag__before_sum__dim = _temp_list.lt(the_input)
    amount__list_gt = _temp__flag__before_sum__dim.sum()
    if not list_is_Ascending:
        amount__list_gt = _temp_list.nelement() - amount__list_gt 
        pass
    start_offset += amount__list_gt
    del amount__list_gt# just in case
    
    if start_offset == 0:
        return torch.tensor(0., device=the_list.device)
    if start_offset == n_element:
        return torch.tensor(n_element-1., device=the_list.device)
    
    fraction = reverse_interpolation(the_list[start_offset-1], the_list[start_offset], the_input, clamp=True)
    return fraction + start_offset -1

if "unused code for a level 2 version" and False:

    def _reverse_interpolation___perf_test_level_2(the_list:torch.Tensor, list_is_Ascending:bool, the_input:torch.Tensor, 
                                                level_big:int, level_small:int, 
                                    Im_sure_the_list_is_sorted = False, )->torch.Tensor:
        '''the list must be sorted!!!
        
        This version is designed for gpu. But the cpu version is not finished. So use this for all cases at the moment.'''
        if not Im_sure_the_list_is_sorted:
            if list_is_Ascending:
                assert the_list[:-1].le(the_list[1:]).all()
                pass
            else:#decending
                assert the_list[:-1].ge(the_list[1:]).all()
                pass
            pass
        
        n_element = the_list.shape[0]
        
        start_offset = 0
        _1_lshift_16 = 1<<level_big
        if n_element>_1_lshift_16:
            _temp_list = the_list[_1_lshift_16::_1_lshift_16]
            _temp__flag__before_sum__dim = _temp_list.lt(the_input)
            amount__list_gt = _temp__flag__before_sum__dim.sum()
            if not list_is_Ascending:
                amount__list_gt = _temp_list.nelement() - amount__list_gt
                pass
            start_offset += amount__list_gt * _1_lshift_16
            pass
        _1_lshift_8 = 1<<level_small
        if n_element>_1_lshift_8:
            _temp_list = the_list[start_offset+_1_lshift_8 : start_offset+_1_lshift_16 : _1_lshift_8]
            _temp__flag__before_sum__dim = _temp_list.lt(the_input)
            amount__list_gt = _temp__flag__before_sum__dim.sum()
            if not list_is_Ascending:
                amount__list_gt = _temp_list.nelement() - amount__list_gt
                pass
            start_offset += amount__list_gt * _1_lshift_8
            pass
        
        
        _temp_list = the_list[start_offset:start_offset+_1_lshift_8]
        _temp__flag__before_sum__dim = _temp_list.lt(the_input)
        amount__list_gt = _temp__flag__before_sum__dim.sum()
        if not list_is_Ascending:
            amount__list_gt = _temp_list.nelement() - amount__list_gt 
            pass
        start_offset += amount__list_gt
        del amount__list_gt# just in case
        
        if start_offset == 0:
            return torch.tensor(0., device=the_list.device)
        if start_offset == n_element:
            return torch.tensor(n_element-1., device=the_list.device)
        
        fraction = reverse_interpolation(the_list[start_offset-1], the_list[start_offset], the_input, clamp=True)
        return fraction + start_offset -1

    pass

def _reverse_interpolation_of_list__parallel__list_must_sorted(the_list:torch.Tensor, list_is_Ascending:bool, the_input:torch.Tensor, 
                                Im_sure_the_list_is_sorted = False, )->torch.Tensor:
    '''the list must be sorted!!!
    
    This version is designed for gpu. But the cpu version is not finished. So use this for all cases at the moment.'''
    if not Im_sure_the_list_is_sorted:
        if list_is_Ascending:
            assert the_list[:-1].le(the_list[1:]).all()
            pass
        else:#decending
            assert the_list[:-1].ge(the_list[1:]).all()
            pass
        pass
    
    n_element = the_list.shape[0]
    
    if n_element<=1<<19:
        return __reverse_interpolation___perf_test_without_level(the_list = the_list, 
                list_is_Ascending = list_is_Ascending, the_input = the_input, 
                Im_sure_the_list_is_sorted = True)
    else:
        return __reverse_interpolation___perf_test_level_1(the_list = the_list, 
                list_is_Ascending = list_is_Ascending, the_input = the_input, 
                level = 15, # based on tested. this is 15.
                Im_sure_the_list_is_sorted = True)
        pass
    pass# end of function.

if "performance test of gpu versions"  and __DEBUG_ME__() and False:
    def ____test____performance_test_of_gpu_versions():
    
        #  1<<n =  [ 15,      17,      19,      20,      21,      22,      23,      24]
        # _22_ = [ 0.0255,  0.0276,  0.0297,  0.0378,  0.0639,  0.0966,  0.0967,  0.1096], min 0.0255
        # _20_ = [ 0.0242,  0.0278,  0.0290,  0.0366,  0.0485,  0.0467,  0.0547,  0.0468], min 0.0242
        # _19_ = [ 0.0264,  0.0276,  0.0286,  0.0396,  0.0403,  0.0472,  0.0379,  0.0379], min 0.0264
        # _no_ = [ 0.0262,  0.0253,  0.0297,  0.0367,  0.0553,  0.0918,  0.1619,  0.2985], min 0.0253
        # _18_ = [ 0.0254,  0.0287,  0.0347,  0.0354,  0.0351,  0.0389,  0.0351,  0.0340], min 0.0254
        # _16_ = [ 0.0262,  0.0379,  0.0363,  0.0330,  0.0326,  0.0330,  0.0333,  0.0327], min 0.0262
        # _15_ = [ 0.0242,  0.0337,  0.0356,  0.0347,  0.0353,  0.0359,  0.0338,  0.0341], min 0.0242
        # _14_ = [ 0.0382,  0.0354,  0.0349,  0.0349,  0.0335,  0.0350,  0.0342,  0.0342], min 0.0335
        # _13_ = [ 0.0378,  0.0354,  0.0364,  0.0345,  0.0353,  0.0342,  0.0331,  0.0333], min 0.0331
        # _12_ = [ 0.0375,  0.0398,  0.0351,  0.0337,  0.0346,  0.0341,  0.0350,  0.0336], min 0.0336
        # final= [ 0.0265,  0.0239,  0.0289,  0.0325,  0.0339,  0.0350,  0.0338,  0.0323], min 0.0239
        # min  = [ 0.0242,  0.0239,  0.0286,  0.0325,  0.0326,  0.0330,  0.0331,  0.0323]
        
        
        
        from pytorch_yagaodirac_v2.timeit_yagaodirac import timeit
        time_at_most = 2.
        loop_time = 100
        
        _20_10_time__list= []
        _18_9_time__list = []
        _16_8_time__list = []
        _14_7_time__list = []
        _12_6_time__list = []
        _32_time__list   = []
        _22_time__list   = []
        _20_time__list   = []
        _19_time__list   = [] 
        _18_time__list   = []
        _16_time__list   = []
        _15_time__list   = []
        _14_time__list   = []
        _13_time__list   = []
        _12_time__list   = []
        _10_time__list   = []
        _8_time__list    = []
        _6_time__list    = []
        _no_time__list   = []
        _final_time__list= []
        
        #n_list = [7,9,11,13,15,17,19,20,21,22,23,24]
        n_list = [15,17,19,20,21,22,23,24]
        n_element_list = []
        for n in n_list:
            n_element_list.append(1<<n)
            pass
        
        for n_element in n_element_list:
            print(n_element)
            
            the_list_gpu = torch.arange(n_element, device='cuda')
            
            #<  warm up
            for _ in range(10):
                the_input = (torch.rand(size=[], device='cuda')*0.8+0.1)*n_element
            #</ warm up
            
            def _timeit_null__gpu():
                for _ in range(loop_time):
                    the_input = (torch.rand(size=[], device='cuda')*0.8+0.1)*n_element
                    ###################################################################
                        ##############################################         ###########################
                        ################################
                    pass
                return
            null_time__gpu = timeit(_timeit_null__gpu, time_at_most=time_at_most)[0]
            del _timeit_null__gpu
            
            #<  warm up
            for _ in range(10):
                the_input = (torch.rand(size=[], device='cuda')*0.8+0.1)*n_element
                _reverse_interpolation___perf_test_level_2(the_list = the_list_gpu,
                    list_is_Ascending = True,the_input = the_input,         level_big=12, level_small=6,
                    Im_sure_the_list_is_sorted = True)
            #</ warm up

            if "double levels" and False:
                
                def _timeit_paral___20_10():
                    for _ in range(loop_time):
                        the_input = (torch.rand(size=[], device='cuda')*0.8+0.1)*n_element
                        _reverse_interpolation___perf_test_level_2(the_list = the_list_gpu,
                            list_is_Ascending = True,the_input = the_input,         level_big=20, level_small=10,
                            Im_sure_the_list_is_sorted = True)
                        pass
                    return
                _raw_20_10_time = timeit(_timeit_paral___20_10, time_at_most=time_at_most)[0]
                _20_10_time = _raw_20_10_time - null_time__gpu
                _20_10_time__list.append(_20_10_time)
                del _timeit_paral___20_10, _raw_20_10_time, _20_10_time
                
                def _timeit_paral___18_9():
                    for _ in range(loop_time):
                        the_input = (torch.rand(size=[], device='cuda')*0.8+0.1)*n_element
                        _reverse_interpolation___perf_test_level_2(the_list = the_list_gpu,
                            list_is_Ascending = True,the_input = the_input,         level_big=18, level_small=9,
                            Im_sure_the_list_is_sorted = True)
                        pass
                    return
                _raw_18_9_time = timeit(_timeit_paral___18_9, time_at_most=time_at_most)[0]
                _18_9_time = _raw_18_9_time - null_time__gpu
                _18_9_time__list.append(_18_9_time)
                del _timeit_paral___18_9, _raw_18_9_time, _18_9_time
                
                def _timeit_paral___16_8():
                    for _ in range(loop_time):
                        the_input = (torch.rand(size=[], device='cuda')*0.8+0.1)*n_element
                        _reverse_interpolation___perf_test_level_2(the_list = the_list_gpu,
                            list_is_Ascending = True,the_input = the_input,         level_big=16, level_small=8,
                            Im_sure_the_list_is_sorted = True)
                        pass
                    return
                _raw_16_8_time = timeit(_timeit_paral___16_8, time_at_most=time_at_most)[0]
                _16_8_time = _raw_16_8_time - null_time__gpu
                _16_8_time__list.append(_16_8_time)
                del _timeit_paral___16_8, _raw_16_8_time, _16_8_time
                
                def _timeit_paral___14_7():
                    for _ in range(loop_time):
                        the_input = (torch.rand(size=[], device='cuda')*0.8+0.1)*n_element
                        _reverse_interpolation___perf_test_level_2(the_list = the_list_gpu,
                            list_is_Ascending = True,the_input = the_input,         level_big=14, level_small=7,
                            Im_sure_the_list_is_sorted = True)
                        pass
                    return
                _raw_14_7_time = timeit(_timeit_paral___14_7, time_at_most=time_at_most)[0]
                _14_7_time = _raw_14_7_time - null_time__gpu
                _14_7_time__list.append(_14_7_time)
                del _timeit_paral___14_7, _raw_14_7_time, _14_7_time
                
                
                def _timeit_paral___12_6():
                    for _ in range(loop_time):
                        the_input = (torch.rand(size=[], device='cuda')*0.8+0.1)*n_element
                        _reverse_interpolation___perf_test_level_2(the_list = the_list_gpu,
                            list_is_Ascending = True,the_input = the_input,         level_big=12, level_small=6,
                            Im_sure_the_list_is_sorted = True)
                        pass
                    return
                _raw_12_6_time = timeit(_timeit_paral___12_6, time_at_most=time_at_most)[0]
                _12_6_time = _raw_12_6_time - null_time__gpu
                _12_6_time__list.append(_12_6_time)
                del _timeit_paral___12_6, _raw_12_6_time, _12_6_time
                
                pass
            
            
            #<  warm up>
            for _ in range(10):
                the_input = (torch.rand(size=[], device='cuda')*0.8+0.1)*n_element
                __reverse_interpolation___perf_test_level_1(the_list = the_list_gpu,
                    list_is_Ascending = True,the_input = the_input,         level = 12,
                    Im_sure_the_list_is_sorted = True)
            #</ warm up>
            
            
            def _timeit_paral___32():
                for _ in range(loop_time):
                    the_input = (torch.rand(size=[], device='cuda')*0.8+0.1)*n_element
                    __reverse_interpolation___perf_test_level_1(the_list = the_list_gpu,
                        list_is_Ascending = True,the_input = the_input,         level = 32,
                        Im_sure_the_list_is_sorted = True)
                    pass
                return
            _raw_32_time = timeit(_timeit_paral___32, time_at_most=time_at_most)[0]
            _32_time = _raw_32_time - null_time__gpu
            _32_time__list.append(_32_time)
            del _timeit_paral___32, _raw_32_time, _32_time
            
            
            
            def _timeit_paral___22():
                for _ in range(loop_time):
                    the_input = (torch.rand(size=[], device='cuda')*0.8+0.1)*n_element
                    __reverse_interpolation___perf_test_level_1(the_list = the_list_gpu,
                        list_is_Ascending = True,the_input = the_input,         level = 22,
                        Im_sure_the_list_is_sorted = True)
                    pass
                return
            _raw_22_time = timeit(_timeit_paral___22, time_at_most=time_at_most)[0]
            _22_time = _raw_22_time - null_time__gpu
            _22_time__list.append(_22_time)
            del _timeit_paral___22, _raw_22_time, _22_time
            
            def _timeit_paral___20():
                for _ in range(loop_time):
                    the_input = (torch.rand(size=[], device='cuda')*0.8+0.1)*n_element
                    __reverse_interpolation___perf_test_level_1(the_list = the_list_gpu,
                        list_is_Ascending = True,the_input = the_input,         level = 20,
                        Im_sure_the_list_is_sorted = True)
                    pass
                return
            _raw_20_time = timeit(_timeit_paral___20, time_at_most=time_at_most)[0]
            _20_time = _raw_20_time - null_time__gpu
            _20_time__list.append(_20_time)
            del _timeit_paral___20, _raw_20_time, _20_time
            
            
            
            def _timeit_paral___19():
                for _ in range(loop_time):
                    the_input = (torch.rand(size=[], device='cuda')*0.8+0.1)*n_element
                    __reverse_interpolation___perf_test_level_1(the_list = the_list_gpu,
                        list_is_Ascending = True,the_input = the_input,         level = 19,
                        Im_sure_the_list_is_sorted = True)
                    pass
                return
            _raw_19_time = timeit(_timeit_paral___19, time_at_most=time_at_most)[0]
            _19_time = _raw_19_time - null_time__gpu
            _19_time__list.append(_19_time)
            del _timeit_paral___19, _raw_19_time, _19_time
            
            def _timeit_paral___18():
                for _ in range(loop_time):
                    the_input = (torch.rand(size=[], device='cuda')*0.8+0.1)*n_element
                    __reverse_interpolation___perf_test_level_1(the_list = the_list_gpu,
                        list_is_Ascending = True,the_input = the_input,         level = 18,
                        Im_sure_the_list_is_sorted = True)
                    pass
                return
            _raw_18_time = timeit(_timeit_paral___18, time_at_most=time_at_most)[0]
            _18_time = _raw_18_time - null_time__gpu
            _18_time__list.append(_18_time)
            del _timeit_paral___18, _raw_18_time, _18_time
            
            def _timeit_paral___16():
                for _ in range(loop_time):
                    the_input = (torch.rand(size=[], device='cuda')*0.8+0.1)*n_element
                    __reverse_interpolation___perf_test_level_1(the_list = the_list_gpu,
                        list_is_Ascending = True,the_input = the_input,         level = 16,
                        Im_sure_the_list_is_sorted = True)
                    pass
                return
            _raw_16_time = timeit(_timeit_paral___16, time_at_most=time_at_most)[0]
            _16_time = _raw_16_time - null_time__gpu
            _16_time__list.append(_16_time)
            del _timeit_paral___16, _raw_16_time, _16_time
            
            
            
            
            
            def _timeit_paral___15():
                for _ in range(loop_time):
                    the_input = (torch.rand(size=[], device='cuda')*0.8+0.1)*n_element
                    __reverse_interpolation___perf_test_level_1(the_list = the_list_gpu,
                        list_is_Ascending = True,the_input = the_input,         level = 15,
                        Im_sure_the_list_is_sorted = True)
                    pass
                return
            _raw_15_time = timeit(_timeit_paral___15, time_at_most=time_at_most)[0]
            _15_time = _raw_15_time - null_time__gpu
            _15_time__list.append(_15_time)
            del _timeit_paral___15, _raw_15_time, _15_time
            
            def _timeit_paral___14():
                for _ in range(loop_time):
                    the_input = (torch.rand(size=[], device='cuda')*0.8+0.1)*n_element
                    __reverse_interpolation___perf_test_level_1(the_list = the_list_gpu,
                        list_is_Ascending = True,the_input = the_input,         level = 14,
                        Im_sure_the_list_is_sorted = True)
                    pass
                return
            _raw_14_time = timeit(_timeit_paral___14, time_at_most=time_at_most)[0]
            _14_time = _raw_14_time - null_time__gpu
            _14_time__list.append(_14_time)
            del _timeit_paral___14, _raw_14_time, _14_time
            
            def _timeit_paral___13():
                for _ in range(loop_time):
                    the_input = (torch.rand(size=[], device='cuda')*0.8+0.1)*n_element
                    __reverse_interpolation___perf_test_level_1(the_list = the_list_gpu,
                        list_is_Ascending = True,the_input = the_input,         level = 13,
                        Im_sure_the_list_is_sorted = True)
                    pass
                return
            _raw_13_time = timeit(_timeit_paral___13, time_at_most=time_at_most)[0]
            _13_time = _raw_13_time - null_time__gpu
            _13_time__list.append(_13_time)
            del _timeit_paral___13, _raw_13_time, _13_time
            
            def _timeit_paral___12():
                for _ in range(loop_time):
                    the_input = (torch.rand(size=[], device='cuda')*0.8+0.1)*n_element
                    __reverse_interpolation___perf_test_level_1(the_list = the_list_gpu,
                        list_is_Ascending = True,the_input = the_input,         level = 12,
                        Im_sure_the_list_is_sorted = True)
                    pass
                return
            _raw_12_time = timeit(_timeit_paral___12, time_at_most=time_at_most)[0]
            _12_time = _raw_12_time - null_time__gpu
            _12_time__list.append(_12_time)
            del _timeit_paral___12, _raw_12_time, _12_time
            
            if False:
                def _timeit_paral___10():
                    for _ in range(loop_time):
                        the_input = (torch.rand(size=[], device='cuda')*0.8+0.1)*n_element
                        __reverse_interpolation___perf_test_level_1(the_list = the_list_gpu,
                            list_is_Ascending = True,the_input = the_input,         level = 10,
                            Im_sure_the_list_is_sorted = True)
                        pass
                    return
                _raw_10_time = timeit(_timeit_paral___10, time_at_most=time_at_most)[0]
                _10_time = _raw_10_time - null_time__gpu
                _10_time__list.append(_10_time)
                del _timeit_paral___10, _raw_10_time, _10_time
                
                def _timeit_paral___8():
                    for _ in range(loop_time):
                        the_input = (torch.rand(size=[], device='cuda')*0.8+0.1)*n_element
                        __reverse_interpolation___perf_test_level_1(the_list = the_list_gpu,
                            list_is_Ascending = True,the_input = the_input,         level = 8,
                            Im_sure_the_list_is_sorted = True)
                        pass
                    return
                _raw_8_time = timeit(_timeit_paral___8, time_at_most=time_at_most)[0]
                _8_time = _raw_8_time - null_time__gpu
                _8_time__list.append(_8_time)
                del _timeit_paral___8, _raw_8_time, _8_time
                
                def _timeit_paral___6():
                    for _ in range(loop_time):
                        the_input = (torch.rand(size=[], device='cuda')*0.8+0.1)*n_element
                        __reverse_interpolation___perf_test_level_1(the_list = the_list_gpu,
                            list_is_Ascending = True,the_input = the_input,         level = 6,
                            Im_sure_the_list_is_sorted = True)
                        pass
                    return
                _raw_6_time = timeit(_timeit_paral___6, time_at_most=time_at_most)[0]
                _6_time = _raw_6_time - null_time__gpu
                _6_time__list.append(_6_time)
                del _timeit_paral___6, _raw_6_time, _6_time
                
                pass
            
            
            
            #<  warm up
            for _ in range(10):
                the_input = (torch.rand(size=[], device='cuda')*0.8+0.1)*n_element
                __reverse_interpolation___perf_test_without_level(the_list = the_list_gpu,
                    list_is_Ascending = True,the_input = the_input,
                    Im_sure_the_list_is_sorted = True)
                pass
            #</ warm up
            
            def _timeit_paral___no():
                for _ in range(loop_time):
                    the_input = (torch.rand(size=[], device='cuda')*0.8+0.1)*n_element
                    __reverse_interpolation___perf_test_without_level(the_list = the_list_gpu,
                        list_is_Ascending = True,the_input = the_input,
                        Im_sure_the_list_is_sorted = True)
                    pass
                return
            _raw_no_time = timeit(_timeit_paral___no, time_at_most=time_at_most)[0]
            _no_time = _raw_no_time - null_time__gpu
            _no_time__list.append(_no_time)
            del _timeit_paral___no, _raw_no_time, _no_time
            
            
            
            
            def _timeit_paral___final():
                for _ in range(loop_time):
                    the_input = (torch.rand(size=[], device='cuda')*0.8+0.1)*n_element
                    _reverse_interpolation_of_list__parallel__list_must_sorted(the_list = the_list_gpu,
                        list_is_Ascending = True,the_input = the_input,
                        Im_sure_the_list_is_sorted = True)
                    pass
                return
            _raw_final_time = timeit(_timeit_paral___final, time_at_most=time_at_most)[0]
            _final_time = _raw_final_time - null_time__gpu
            _final_time__list.append(_final_time)
            del _timeit_paral___final, _raw_final_time, _final_time
            #-----------------------#-----------------------#-----------------------
            
            pass
        
        print(f" 1<<n =  {str_the_list(n_list, 0, segment=",     ")}")
        
        # print(f"_20_10  {str_the_list(_20_10_time__list , 4)}")
        # print(f"_18_9_  {str_the_list(_18_9_time__list  , 4)}")
        # print(f"_16_8_  {str_the_list(_16_8_time__list  , 4)}")
        # print(f"_14_7_  {str_the_list(_14_7_time__list  , 4)}")
        # print(f"_12_6_  {str_the_list(_12_6_time__list  , 4)}")
        #print(f"_32_ti  {str_the_list(_32_time__list    , 4)}")#worse than without leve.
        print(f"_22_ = {str_the_list(_22_time__list    , 4)}, min {min(_22_time__list    ):.4f}")
        print(f"_20_ = {str_the_list(_20_time__list    , 4)}, min {min(_20_time__list    ):.4f}")
        print(f"_19_ = {str_the_list(_19_time__list    , 4)}, min {min(_19_time__list    ):.4f}")
        print(f"_no_ = {str_the_list(_no_time__list    , 4)}, min {min(_no_time__list    ):.4f}")
        print(f"_18_ = {str_the_list(_18_time__list    , 4)}, min {min(_18_time__list    ):.4f}")
        print(f"_16_ = {str_the_list(_16_time__list    , 4)}, min {min(_16_time__list    ):.4f}")
        print(f"_15_ = {str_the_list(_15_time__list    , 4)}, min {min(_15_time__list    ):.4f}")
        print(f"_14_ = {str_the_list(_14_time__list    , 4)}, min {min(_14_time__list    ):.4f}")
        print(f"_13_ = {str_the_list(_13_time__list    , 4)}, min {min(_13_time__list    ):.4f}")
        print(f"_12_ = {str_the_list(_12_time__list    , 4)}, min {min(_12_time__list    ):.4f}")
        print(f"final= {str_the_list(_final_time__list , 4)}, min {min(_final_time__list ):.4f}")
        
        _temp_tensor = torch.tensor([
            _22_time__list    ,
            _20_time__list    ,
            _19_time__list    ,
            _no_time__list    ,
            _18_time__list    ,
            _16_time__list    ,
            _15_time__list    ,
            _14_time__list    ,
            _13_time__list    ,
            _12_time__list    ,
            _final_time__list ,])
        bottom_line = _temp_tensor.min(dim=0).values
        
        print(f"min  = {str_the_list(bottom_line , 4)}")
        
        # print(f"_10_ti  {str_the_list(_10_time__list    , 4)}")
        # print(f"_8_tim  {str_the_list(_8_time__list     , 4)}")
        # print(f"_6_tim  {str_the_list(_6_time__list     , 4)}")
        
        return 
    
    ____test____performance_test_of_gpu_versions()
    pass



if "test" and __DEBUG_ME__() and False:
    def ____test____reverse_interpolation_of_list():
        
        if "ascending" and True:
            #long list
            the_list = torch.tensor([12.,16,20,30])
            result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(12.))
            assert _tensor_equal(result, [0.])
            result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(11.))
            assert _tensor_equal(result, [0.])
            result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(30.))
            assert _tensor_equal(result, [3.])
            result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(31.))
            assert _tensor_equal(result, [3.])
            
            result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(14.))
            assert _tensor_equal(result, [0.5])
            result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(13.))
            assert _tensor_equal(result, [0.25])
            result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(17.))
            assert _tensor_equal(result, [1.25])
            result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(28.))
            assert _tensor_equal(result, [2.8])
            
            # around 256?
            the_list = torch.arange(1000, dtype=torch.float32)
            assert the_list.nelement() == 1000
            result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(0.3))
            assert _tensor_equal(result, [0.3])
            result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(1.3))
            assert _tensor_equal(result, [1.3])
            for ii in range(999):
                result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(ii+0.))
                assert _tensor_equal(result, [ii+0.])
                pass
            result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(998.))
            assert _tensor_equal(result, [998.])
            result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(999.))
            assert _tensor_equal(result, [999.])
            result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(999.5))
            assert _tensor_equal(result, [999.])
            
            the_list = torch.arange(1000, dtype=torch.float32)+2.
            assert the_list.nelement() == 1000
            result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(2.3))
            assert _tensor_equal(result, [0.3])
            result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(3.3))
            assert _tensor_equal(result, [1.3])
            for ii in range(999):
                result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(ii+2.))
                assert _tensor_equal(result, [ii+0.])
                pass
            result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(1000.))
            assert _tensor_equal(result, [998.])
            result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(1001.))
            assert _tensor_equal(result, [999.])
            result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(1001.5))
            assert _tensor_equal(result, [999.])
            
            
            # around 65536
            the_list = torch.arange(70000, dtype=torch.float32)
            assert the_list.nelement() == 70000
            result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(0.3))
            assert _tensor_equal(result, [0.3])
            for ii in range(65500, 65700):
                result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(ii+0.))
                assert _tensor_equal(result, [ii+0.])
                pass
            for ii in range(65500+256, 65700+256):
                result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(ii+0.))
                assert _tensor_equal(result, [ii+0.])
                pass
            for ii in range(65500+512, 65700+512):
                result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(ii+0.))
                assert _tensor_equal(result, [ii+0.])
                pass
            
            the_list = torch.arange(70000, dtype=torch.float32)+2.
            assert the_list.nelement() == 70000
            result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(2.3))
            assert _tensor_equal(result, [0.3])
            for ii in range(65500, 65700):
                result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(ii+2.))
                assert _tensor_equal(result, [ii+0.])
                pass
            for ii in range(65500+256, 65700+256):
                result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(ii+2.))
                assert _tensor_equal(result, [ii+0.])
                pass
            for ii in range(65500+512, 65700+512):
                result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(ii+2.))
                assert _tensor_equal(result, [ii+0.])
                pass
            
            # more around the seperation point.
            for n_element in [256, 257, 258, 65532, 65533, 65534, 65535, 65536, 65537, 65538, ]:
                        #16777212, 16777213, 16777214, 16777215, 16777216, 16777217, 16777218, 16777219]:
                the_list = torch.zeros(size=[n_element])
                the_list[0] = -11.
                the_list[1] = -1.
                the_list[-2] = 1.
                the_list[-1] = 2.
                result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(-13.))
                assert _tensor_equal(result, [0.])
                result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(-11.))
                assert _tensor_equal(result, [0.])
                result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(-2.))
                assert _tensor_equal(result, [0.9])
                
                result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(3.))
                assert _tensor_equal(result, [n_element-1.])
                result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(2.))
                assert _tensor_equal(result, [n_element-1.])
                result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=True, the_input = torch.tensor(1.123))
                assert _tensor_equal(result, [n_element-2.+0.123], epsilon=result*1e-7)
                pass
        
            pass
        
        if "decending" and True:
            #long list
            the_list = torch.tensor([12.,16,20,30])*-1.
            result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(-12.))
            assert _tensor_equal(result, [0.])
            result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(-11.))
            assert _tensor_equal(result, [0.])
            result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(-30.))
            assert _tensor_equal(result, [3.])
            result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(-31.))
            assert _tensor_equal(result, [3.])
            
            result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(-14.))
            assert _tensor_equal(result, [0.5])
            result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(-13.))
            assert _tensor_equal(result, [0.25])
            result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(-17.))
            assert _tensor_equal(result, [1.25])
            result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(-28.))
            assert _tensor_equal(result, [2.8])
            
            # around 256?
            the_list = torch.arange(1000, dtype=torch.float32)*-1.
            assert the_list.nelement() == 1000
            result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(-0.3))
            assert _tensor_equal(result, [0.3])
            result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(-1.3))
            assert _tensor_equal(result, [1.3])
            for ii in range(999):
                result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(-1.*ii))
                assert _tensor_equal(result, [ii+0.])
                pass
            result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(-998.))
            assert _tensor_equal(result, [998.])
            result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(-999.))
            assert _tensor_equal(result, [999.])
            result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(-999.5))
            assert _tensor_equal(result, [999.])
            
            the_list = torch.arange(1000, dtype=torch.float32)*-1.-2.
            assert the_list.nelement() == 1000
            result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(-2.3))
            assert _tensor_equal(result, [0.3])
            result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(-3.3))
            assert _tensor_equal(result, [1.3])
            for ii in range(999):
                result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(-1.*ii-2.))
                assert _tensor_equal(result, [ii+0.])
                pass
            result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(-1000.))
            assert _tensor_equal(result, [998.])
            result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(-1001.))
            assert _tensor_equal(result, [999.])
            result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(-1001.5))
            assert _tensor_equal(result, [999.])
            
            
            # around 65536
            the_list = torch.arange(70000, dtype=torch.float32)*-1.
            assert the_list.nelement() == 70000
            result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(-0.3))
            assert _tensor_equal(result, [0.3])
            for ii in range(65500, 65700):
                result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(-1.*ii))
                assert _tensor_equal(result, [ii+0.])
                pass
            for ii in range(65500+256, 65700+256):
                result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(-1.*ii))
                assert _tensor_equal(result, [ii+0.])
                pass
            for ii in range(65500+512, 65700+512):
                result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(-1.*ii))
                assert _tensor_equal(result, [ii+0.])
                pass
            
            the_list = torch.arange(70000, dtype=torch.float32)*-1.-2.
            assert the_list.nelement() == 70000
            result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(-2.3))
            assert _tensor_equal(result, [0.3])
            for ii in range(65500, 65700):
                result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(-1.*ii-2.))
                assert _tensor_equal(result, [ii+0.])
                pass
            for ii in range(65500+256, 65700+256):
                result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(-1.*ii-2.))
                assert _tensor_equal(result, [ii+0.])
                pass
            for ii in range(65500+512, 65700+512):
                result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(-1.*ii-2.))
                assert _tensor_equal(result, [ii+0.])
                pass
            
            # more around the seperation point.
            for n_element in [256, 257, 258, 65532, 65533, 65534, 65535, 65536, 65537, 65538, ]:
                        #16777212, 16777213, 16777214, 16777215, 16777216, 16777217, 16777218, 16777219]:
                the_list = torch.zeros(size=[n_element])
                the_list[0] = 11.
                the_list[1] = 1.
                the_list[-2] = -1.
                the_list[-1] = -2.
                result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(13.))
                assert _tensor_equal(result, [0.])
                result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(11.))
                assert _tensor_equal(result, [0.])
                result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(2.))
                assert _tensor_equal(result, [0.9])
                
                result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(-3.))
                assert _tensor_equal(result, [n_element-1.])
                result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(-2.))
                assert _tensor_equal(result, [n_element-1.])
                result = _reverse_interpolation_of_list__parallel__list_must_sorted(the_list, list_is_Ascending=False, the_input = torch.tensor(-1.123))
                assert _tensor_equal(result, [n_element-2.+0.123], epsilon=result*1e-7)
                pass
        
            pass
        
        return 
    
    ____test____reverse_interpolation_of_list()
    pass

if "test      are all of them behave the same way???" and __DEBUG_ME__() and False:
    def ____test____3_also_should_behave_the_same():
        
        if "ascending" and True:
            
            the_list = []
            _sum = 2.
            for _ in range(1000):
                _sum += random.randint(1,4)
                the_list.append(_sum)
                pass
            the_list = torch.tensor(the_list)
            the_list = the_list.to(torch.float32)
            
            for ii in range(1010):
                the_input = torch.tensor(ii-5.)
                result_sequencial = _reverse_interpolation_of_list__sequencial__list_must_sorted(the_list = the_list,
                        list_is_Ascending = True,the_input = the_input, Im_sure_the_list_is_sorted = True)
                result_bs =         _reverse_interpolation_of_list__binary_search__list_must_sorted(the_list = the_list,
                        list_is_Ascending = True,the_input = the_input, Im_sure_the_list_is_sorted = True)
                result_parallel =   _reverse_interpolation_of_list__parallel__list_must_sorted(the_list=the_list,
                        list_is_Ascending=True, the_input=the_input, Im_sure_the_list_is_sorted = True)
                
                result_no_level = __reverse_interpolation___perf_test_without_level(the_list=the_list,
                        list_is_Ascending=True, the_input=the_input, Im_sure_the_list_is_sorted = True)
                result_l2 =       __reverse_interpolation___perf_test_level_1(the_list=the_list,
                        list_is_Ascending=True, the_input=the_input, Im_sure_the_list_is_sorted = True, level=2)
                result_l3 =       __reverse_interpolation___perf_test_level_1(the_list=the_list,
                        list_is_Ascending=True, the_input=the_input, Im_sure_the_list_is_sorted = True, level=3)
                result_l4 =       __reverse_interpolation___perf_test_level_1(the_list=the_list,
                        list_is_Ascending=True, the_input=the_input, Im_sure_the_list_is_sorted = True, level=4)
                result_l6 =       __reverse_interpolation___perf_test_level_1(the_list=the_list,
                        list_is_Ascending=True, the_input=the_input, Im_sure_the_list_is_sorted = True, level=6)
                result_l8 =       __reverse_interpolation___perf_test_level_1(the_list=the_list,
                        list_is_Ascending=True, the_input=the_input, Im_sure_the_list_is_sorted = True, level=8)
                
                assert _tensor_equal(result_sequencial, result_bs       )
                assert _tensor_equal(result_sequencial, result_parallel )
                assert _tensor_equal(result_sequencial, result_no_level )
                assert _tensor_equal(result_sequencial, result_l2       )
                assert _tensor_equal(result_sequencial, result_l3       )
                assert _tensor_equal(result_sequencial, result_l4       )
                assert _tensor_equal(result_sequencial, result_l6       )
                assert _tensor_equal(result_sequencial, result_l8       )
                pass
            pass#/ test   if "ascending" 
        
        
        if "decending" and True:
            
            the_list = []
            _sum = 2.
            for _ in range(1000):
                _sum -= random.randint(1,4)
                the_list.append(_sum)
                pass
            the_list = torch.tensor(the_list)
            the_list = the_list.to(torch.float32)
            
            for ii in range(1010):
                the_input = torch.tensor(ii-5.)
                result_sequencial = _reverse_interpolation_of_list__sequencial__list_must_sorted(the_list = the_list,
                        list_is_Ascending = False,the_input = the_input, Im_sure_the_list_is_sorted = True)
                result_bs =         _reverse_interpolation_of_list__binary_search__list_must_sorted(the_list = the_list,
                        list_is_Ascending = False,the_input = the_input, Im_sure_the_list_is_sorted = True)
                result_parallel =   _reverse_interpolation_of_list__parallel__list_must_sorted(the_list=the_list,
                        list_is_Ascending=False, the_input=the_input, Im_sure_the_list_is_sorted = True)
                
                result_no_level = __reverse_interpolation___perf_test_without_level(the_list=the_list,
                        list_is_Ascending=False, the_input=the_input, Im_sure_the_list_is_sorted = True)
                result_l2 =       __reverse_interpolation___perf_test_level_1(the_list=the_list,
                        list_is_Ascending=False, the_input=the_input, Im_sure_the_list_is_sorted = True, level=2)
                result_l3 =       __reverse_interpolation___perf_test_level_1(the_list=the_list,
                        list_is_Ascending=False, the_input=the_input, Im_sure_the_list_is_sorted = True, level=3)
                result_l4 =       __reverse_interpolation___perf_test_level_1(the_list=the_list,
                        list_is_Ascending=False, the_input=the_input, Im_sure_the_list_is_sorted = True, level=4)
                result_l6 =       __reverse_interpolation___perf_test_level_1(the_list=the_list,
                        list_is_Ascending=False, the_input=the_input, Im_sure_the_list_is_sorted = True, level=6)
                result_l8 =       __reverse_interpolation___perf_test_level_1(the_list=the_list,
                        list_is_Ascending=False, the_input=the_input, Im_sure_the_list_is_sorted = True, level=8)
                
                assert _tensor_equal(result_sequencial, result_bs       )
                assert _tensor_equal(result_sequencial, result_parallel )
                assert _tensor_equal(result_sequencial, result_no_level )
                assert _tensor_equal(result_sequencial, result_l2       )
                assert _tensor_equal(result_sequencial, result_l3       )
                assert _tensor_equal(result_sequencial, result_l4       )
                assert _tensor_equal(result_sequencial, result_l6       )
                assert _tensor_equal(result_sequencial, result_l8       )
                pass
            pass#/ test   if "ascending" 
        
        return 
    
    ____test____3_also_should_behave_the_same()
    pass

if "performance test of all 3          等一下。。。"  and __DEBUG_ME__() and False:
    def ____test____performance_test_of_all_3():
        
        if "ascending" and False:
            # n_element     = [ 2,      10,      16,      24,      30,      40,      100,      1000]
            # null_time     = [ 0.0010,  0.0010,  0.0010,  0.0010,  0.0011,  0.0010,  0.0010,  0.0010]
            # seq_time      = [ 0.0011,  0.0034,  0.0046,  0.0062,  0.0076,  0.0094,  0.0220,  0.1956]
            # bs_time       = [ 0.0018,  0.0035,  0.0040,  0.0044,  0.0043,  0.0045,  0.0050,  0.0068]
            # null_time_gpu = [ 0.0040,  0.0040,  0.0050,  0.0046,  0.0042,  0.0042,  0.0041,  0.0040]
            # para_time     = [ 0.0187,  0.0240,  0.0269,  0.0289,  0.0248,  0.0255,  0.0284,  0.0237]
            
            from pytorch_yagaodirac_v2.timeit_yagaodirac import timeit
            time_at_most = 2.
            
            null_time__list = []
            #raw_seq_time__list = []
            seq_time__list = []
            #raw_bs_time__list = []
            bs_time__list = []
            null_time_gpu__list = []
            #raw_para_time__list = []
            para_time__list = []
            
            
            #n_element_list = [2,3,4,8,16,64,256,1000,10000]
            n_element_list = [2,10,16,24,30,40,100,1000]
            n_element_list = [2,10,16,24,30,40,100,1000]
            for n_element in n_element_list:
                print(n_element)
                
                #-----------------------#-----------------------#-----------------------
                the_list = torch.arange(n_element)
                
                def _timeit_null():
                    for _ in range(100):
                        the_input = (torch.rand(size=[])*0.8+0.1)*n_element
                        ################################################################################
                            ###################################################################################
                        pass
                    return
                null_time = timeit(_timeit_null, time_at_most=time_at_most)[0]
                null_time__list.append(null_time)
                del _timeit_null
                
                def _timeit_seq():
                    for _ in range(100):
                        the_input = (torch.rand(size=[])*0.8+0.1)*n_element
                        _reverse_interpolation_of_list__sequencial__list_must_sorted(the_list = the_list,
                            list_is_Ascending = True,the_input = the_input, Im_sure_the_list_is_sorted = True)
                        pass
                    return
                _timeit_seq()# warm up
                raw_seq_time = timeit(_timeit_seq, time_at_most=time_at_most)[0]
                seq_time = raw_seq_time-null_time
                seq_time__list.append(seq_time)
                del _timeit_seq
                
                def _timeit_bs():
                    for _ in range(100):
                        the_input = (torch.rand(size=[])*0.8+0.1)*n_element
                        _reverse_interpolation_of_list__binary_search__list_must_sorted(the_list = the_list,
                            list_is_Ascending = True,the_input = the_input, Im_sure_the_list_is_sorted = True)
                        pass
                    return
                _timeit_bs()# warm up
                raw_bs_time = timeit(_timeit_bs, time_at_most=time_at_most)[0]
                bs_time = raw_bs_time-null_time
                bs_time__list.append(bs_time)
                del _timeit_bs
                del null_time
                
                #<  CPU         CPU         CPU         CPU         CPU         
                #<  GPU         GPU         GPU         GPU         GPU         
                
                the_list_gpu = torch.arange(n_element, device='cuda')
                
                def _timeit_null__gpu():
                    for _ in range(100):
                        the_input = (torch.rand(size=[], device='cuda')*0.8+0.1)*n_element
                        ################################################################################
                            ###################################################################################
                        pass
                    return
                null_time__gpu = timeit(_timeit_null__gpu, time_at_most=time_at_most)[0]
                null_time_gpu__list.append(null_time__gpu)
                del _timeit_null__gpu
                
                def _timeit_paral():
                    for _ in range(100):
                        the_input = (torch.rand(size=[], device='cuda')*0.8+0.1)*n_element
                        _reverse_interpolation_of_list__parallel__list_must_sorted(the_list = the_list_gpu,
                            list_is_Ascending = True,the_input = the_input, Im_sure_the_list_is_sorted = True)
                        pass
                    return
                _timeit_paral()# warm up
                raw_para_time = timeit(_timeit_paral, time_at_most=time_at_most)[0]
                para_time = raw_para_time-null_time__gpu
                para_time__list.append(para_time)
                del _timeit_paral
                #-----------------------#-----------------------#-----------------------
                
                pass
            
            print(f"n_element       = {str_the_list(n_element_list, 0, segment=",     ")}")
            
            print(f"null_time     = {str_the_list(null_time__list , 4)}")
            print(f"seq_time      = {str_the_list(seq_time__list     , 4)}")
            print(f"bs_time       = {str_the_list(bs_time__list      , 4)}")
            print(f"null_time_gpu = {str_the_list(null_time_gpu__list, 4)}")
            print(f"para_time     = {str_the_list(para_time__list    , 4)}")
            
            pass#/ test
        
        if "ascending" and False:
            
            # decending
            # n_element       = [ 2,      10,      16,      24,      30,      40,      100,      1000]
            # null_time     = [ 0.0014,  0.0013,  0.0013,  0.0013,  0.0058,  0.0060,  0.0059,  0.0074]
            # seq_time      = [ 0.0009,  0.0026,  0.0034,  0.0144,  0.0291,  0.0301,  0.0769,  0.7422]
            # bs_time       = [ 0.0013,  0.0027,  0.0029,  0.0197,  0.0157,  0.0155,  0.0195,  0.0233]
            # null_time_gpu = [ 0.0052,  0.0051,  0.0092,  0.0153,  0.0160,  0.0169,  0.0178,  0.0174]
            # para_time     = [ 0.0150,  0.0136,  0.0161,  0.0440,  0.0450,  0.0496,  0.0389,  0.0438]
            
            from pytorch_yagaodirac_v2.timeit_yagaodirac import timeit
            time_at_most = 2.
            
            null_time__list = []
            #raw_seq_time__list = []
            seq_time__list = []
            #raw_bs_time__list = []
            bs_time__list = []
            null_time_gpu__list = []
            #raw_para_time__list = []
            para_time__list = []
            
            
            #n_element_list = [2,3,4,8,16,64,256,1000,10000]
            n_element_list = [2,10,16,24,30,40,100,1000]
            n_element_list = [2,10,16,24,30,40,100,1000]
            for n_element in n_element_list:
                print(n_element)
                
                #-----------------------#-----------------------#-----------------------
                the_list = torch.arange(n_element)*-1.
                
                def _timeit_null():
                    for _ in range(100):
                        the_input = (torch.rand(size=[])*0.8+0.1)*n_element*-1.
                        ################################################################################
                            ###################################################################################
                        pass
                    return
                null_time = timeit(_timeit_null, time_at_most=time_at_most)[0]
                null_time__list.append(null_time)
                del _timeit_null
                
                def _timeit_seq():
                    for _ in range(100):
                        the_input = (torch.rand(size=[])*0.8+0.1)*n_element*-1.
                        _reverse_interpolation_of_list__sequencial__list_must_sorted(the_list = the_list,
                            list_is_Ascending = False,the_input = the_input, Im_sure_the_list_is_sorted = True)
                        pass
                    return
                _timeit_seq()# warm up
                raw_seq_time = timeit(_timeit_seq, time_at_most=time_at_most)[0]
                seq_time = raw_seq_time-null_time
                seq_time__list.append(seq_time)
                del _timeit_seq
                
                def _timeit_bs():
                    for _ in range(100):
                        the_input = (torch.rand(size=[])*0.8+0.1)*n_element*-1.
                        _reverse_interpolation_of_list__binary_search__list_must_sorted(the_list = the_list,
                            list_is_Ascending = False,the_input = the_input, Im_sure_the_list_is_sorted = True)
                        pass
                    return
                _timeit_bs()# warm up
                raw_bs_time = timeit(_timeit_bs, time_at_most=time_at_most)[0]
                bs_time = raw_bs_time-null_time
                bs_time__list.append(bs_time)
                del _timeit_bs
                del null_time
                
                #<  CPU         CPU         CPU         CPU         CPU         
                #<  GPU         GPU         GPU         GPU         GPU         
                
                the_list_gpu = torch.arange(n_element, device='cuda')
                
                def _timeit_null__gpu():
                    for _ in range(100):
                        the_input = (torch.rand(size=[], device='cuda')*0.8+0.1)*n_element*-1.
                        ################################################################################
                            ###################################################################################
                        pass
                    return
                null_time__gpu = timeit(_timeit_null__gpu, time_at_most=time_at_most)[0]
                null_time_gpu__list.append(null_time__gpu)
                del _timeit_null__gpu
                
                def _timeit_paral():
                    for _ in range(100):
                        the_input = (torch.rand(size=[], device='cuda')*0.8+0.1)*n_element*-1.
                        _reverse_interpolation_of_list__parallel__list_must_sorted(the_list = the_list_gpu,
                            list_is_Ascending = False,the_input = the_input, Im_sure_the_list_is_sorted = True)
                        pass
                    return
                _timeit_paral()# warm up
                raw_para_time = timeit(_timeit_paral, time_at_most=time_at_most)[0]
                para_time = raw_para_time-null_time__gpu
                para_time__list.append(para_time)
                del _timeit_paral
                #-----------------------#-----------------------#-----------------------
                
                pass
            
            print(f"decending")
            print(f"n_element       = {str_the_list(n_element_list, 0, segment=",     ")}")
            
            print(f"null_time     = {str_the_list(null_time__list , 4)}")
            print(f"seq_time      = {str_the_list(seq_time__list     , 4)}")
            print(f"bs_time       = {str_the_list(bs_time__list      , 4)}")
            print(f"null_time_gpu = {str_the_list(null_time_gpu__list, 4)}")
            print(f"para_time     = {str_the_list(para_time__list    , 4)}")
            
            pass#/ test
        
        
        return 
    
    ____test____performance_test_of_all_3()
    pass






def reverse_interpolation_of_list__list_must_sorted(the_list:torch.Tensor, list_is_Ascending:bool, the_input:torch.Tensor, 
                            Im_sure_the_list_is_sorted = False)->torch.Tensor:
    '''the list must be sorted!!!'''
    if not Im_sure_the_list_is_sorted:
        if list_is_Ascending:
            assert the_list[:-1].le(the_list[1:]).all()
            pass
        else:#decending
            assert the_list[:-1].ge(the_list[1:]).all()
            pass
        pass
    
    n_element = the_list.shape[0]
    
    if the_list.device.type == 'cpu':
        if n_element<=10:
            return _reverse_interpolation_of_list__sequencial__list_must_sorted(the_list = the_list,
                    list_is_Ascending = list_is_Ascending, the_input = the_input, 
                    Im_sure_the_list_is_sorted = True)
        else:
            return _reverse_interpolation_of_list__binary_search__list_must_sorted(the_list = the_list,
                    list_is_Ascending = list_is_Ascending, the_input = the_input, 
                    Im_sure_the_list_is_sorted = True)
        pass
    elif the_list.device.type == 'cuda':
        return _reverse_interpolation_of_list__parallel__list_must_sorted(the_list = the_list,
                    list_is_Ascending = list_is_Ascending, the_input = the_input, 
                    Im_sure_the_list_is_sorted = True)
        pass
    else:
        assert False, "unknown device. Implement it yourself, or steer the branch to any of my versions."
    # end of function.


