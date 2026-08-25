import time
from typing import List, Tuple, Optional, Literal
import torch
import math, random
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
#from pytorch_yagaodirac_v2.timeit_yagaodirac import timeit
#from pytorch_yagaodirac_v2.Util import 

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









"log10 tool"

# Here I provide 2 versions.
# the old version is on top. It's better but not a pure gpu implementation.
# Then I made a new torch-specified version. It's faster.
# if you write code on other tool, I recommend shift the old version, bc the also is better.
# but if you write on pytorch, use the new version.
# the behavior and results are slightly different.

# I did tests in log10 measurement.py file. Check it out if you are learning this tool.

def get_mask_of_top_element__rough(input__b_i:torch.Tensor, top_ratio = 0.9, error_of_ratio__at_least = 0.01, \
                            bottom = False, careful_level:int|None = 3, epsilon__for_binary_search:float|torch.Tensor|None = None, \
                    _needs_log__before_loop = False, _needs_log__basic_of_loop = False, \
                    _needs_log__binary_search_in_loop = False, \
                    _needs_log__error_ratio_in_loop = False, \
                                    )->tuple[torch.Tensor, dict[str, list[str]]|None]:
    ''' 
    return _temp_tensor, _log
    return _temp_tensor, _log
    return _temp_tensor, _log
    return _temp_tensor, _log
    
    重新整理一下思路
    这个函数有2个退出模式。
    1是，准确的找到了所需要的比例。比例的目标区间本身是越来越宽的。
    2是，类似二分查找的那种上下限，如果距离足够近，也就退出了。
    核心思想就是，
    目标，和error，算出允许的比例的上下限  top_ratio  error   at_least/most_this_amount 
    二分查找的那个标准是最大值和最小值的平均值，然后根据需要的方向来缩小。   max, min,(threshold), epi
    
    maintainance note:
    this function is init+loop, the loop is binary_search+if_return+if_repeating
    or, it's
    def ...():
        init
        while true:
            binary_search
            if_return?(the only return)
            if_repeating
            pass#while true
        pass#end of function
    In init, there's a bit early return. I didn't remove it, but it should not be triggered.
    The binary search is not an exact binary search, but it feels similar. The max or min is assigned with mid 
        according to the condition, to modify the threshold.
    Then test if the return condition is met.
    The if_repeating is something detecting if the loop is too repeating. If so, it broadens the error_of_ratio,
        and makes the return condition looser.
    The function guaruntees to return nomatter what you provide, unless there's inf,-inf or nan. 
    I didn't test this extreme case. So please make sure the input is legit numbers.
    
    Shape:
    
    Input is [batch, something], return shape is the same as input. dtype of return is torch.bool.
    
    Dimention name:
    
    shape is [*B*atch, *I*nput]
    
    if shape is too small, this may not work. Valid code uses 3 or 5 elements/batch, but this function is design 
        to process >10 elements/batch input.
    '''
    assert input__b_i.shape.__len__() == 2
    assert top_ratio>0.
    assert top_ratio<1.
    assert error_of_ratio__at_least>=0.
    if careful_level is None:
        careful_level = 3
    assert careful_level>0
    assert careful_level<64, "or modify the data type. search for repeating__b = torch.zeros_like(_the_max_to_calc_threshold__b, dtype=torch.int8)"
    
    _cpu = input__b_i.device.type != "cuda"
    if _cpu and _needs_log__before_loop:
        _log:dict[str, list[str]]|None = {}
        pass
    else:
        _log = None  
        pass
    
    if epsilon__for_binary_search:
        if isinstance(epsilon__for_binary_search, float):
            epsilon__for_binary_search__s = torch.tensor(epsilon__for_binary_search, device=input__b_i.device, dtype=input__b_i.dtype)
            pass
        else:
            # if the assertions don't pass, modify it. I didn't test it very carefully. 
            # simply reference the line above.
            assert isinstance(epsilon__for_binary_search, torch.Tensor)
            assert epsilon__for_binary_search.dtype == input__b_i.dtype
            epsilon__for_binary_search__s = epsilon__for_binary_search.clone().to(input__b_i.device)
            if epsilon__for_binary_search.shape == torch.Size([]):
                epsilon__for_binary_search__s = epsilon__for_binary_search__s.reshape([1,])
                pass
            assert epsilon__for_binary_search__s.shape == torch.Size([1])
        pass
    if (_log is not None) and _needs_log__before_loop:
        _log["before_loop"] = [f"epsilon__for_binary_search:{epsilon__for_binary_search}"]
        pass
    
    
    #dtype uint
    #best dtype for count the amount.
    _element_per_batch__s = input__b_i.shape[1]
    # device = input.device
    # param_factory = {"device":device, "dtype":dtype}
    #dtype int
    if _element_per_batch__s<=(1<<6):
        int_dtype = torch.int8
        pass
    elif _element_per_batch__s<=(1<<14):
        int_dtype = torch.int16
        pass
    elif _element_per_batch__s<=(1<<30):
        int_dtype = torch.int32
        pass
    else:
        int_dtype = torch.int64
        pass
    if _log and _needs_log__before_loop:
        _log["before_loop"].append(f"int type:{str(int_dtype)}")
        pass
    
    with torch.no_grad():
        #into torch.
        careful_level__s:torch.Tensor = torch.tensor(careful_level, device=input__b_i.device)
        del careful_level
        if bottom:
            top_ratio = 1.- top_ratio
            pass
        top_ratio__s = torch.tensor(top_ratio, dtype=torch.float64, device=input__b_i.device)
        del top_ratio
        if _log and _needs_log__before_loop:
            _log["before_loop"].append(f"top ratio:{top_ratio__s}, is bottom:{bottom}")
            pass
        
        #init error_of_ratio 
        better_error_of_ratio = 0.501/_element_per_batch__s
        if better_error_of_ratio<error_of_ratio__at_least:
            better_error_of_ratio = error_of_ratio__at_least
            pass
        del error_of_ratio__at_least
        error_of_ratio__b = torch.empty(size=[input__b_i.shape[0]], device=input__b_i.device)#.reshape([-1,1])
        error_of_ratio__b.fill_(better_error_of_ratio)
        if _log and _needs_log__before_loop:
            _log["before_loop"].append(f"error_of_ratio__b init to:{error_of_ratio__b}")
            pass
        
        #ratio+-error, this segment appears twice in this function.
        at_least_this_amount__b = ((_element_per_batch__s-2)*(top_ratio__s - error_of_ratio__b)+1.4999).to(int_dtype)#.reshape([-1,1])
        at_most_this_amount__b =  ((_element_per_batch__s-2)*(top_ratio__s + error_of_ratio__b)+1.5001).to(int_dtype)#.reshape([-1,1])
        if _log and _needs_log__before_loop:
            _log["before_loop"].append(f"at_least_this_amount__b init to:{at_least_this_amount__b}")
            _log["before_loop"].append(f"at_most_this_amount__b init to:{at_most_this_amount__b}")
            pass
        
        #safety, or maybe a early return.
        _flag_all_true_early_return__b = at_least_this_amount__b.ge(_element_per_batch__s)
        if _flag_all_true_early_return__b.all():
            _temp_tensor = torch.ones_like(input__b_i, dtype=torch.bool, device=input__b_i.device)
            if _log and _needs_log__before_loop:
                _log["before_loop"].append(f"_flag_all_true_early_return__b:{_flag_all_true_early_return__b}, all true, [return]")
                pass
            return _temp_tensor, _log
        _flag_all_true_early_return__b = at_most_this_amount__b.le(0)
        if _flag_all_true_early_return__b.all():
            _temp_tensor = torch.zeros_like(input__b_i, dtype=torch.bool, device=input__b_i.device)
            if _log and _needs_log__before_loop:
                _log["before_loop"].append(f"_flag_all_true_early_return__b:{_flag_all_true_early_return__b}, all true, [return]")
                pass
            return _temp_tensor, _log
        
        #maybe optimizable. reverse+reverse = nothing.
        if_finished__b = (_flag_all_true_early_return__b).logical_or(_flag_all_true_early_return__b)
        if _log and _needs_log__before_loop:
            _log["before_loop"].append(f"if_finished__b init to:{if_finished__b}")
            pass
        
        # init before loop
        _the_max_to_calc_threshold__b:torch.Tensor = input__b_i.max(dim=1).values#.reshape([-1,1])#111111111111111111
        _the_min_to_calc_threshold__b:torch.Tensor = input__b_i.min(dim=1).values#.reshape([-1,1])
        if input__b_i.dtype != torch.float64 and input__b_i.dtype != torch.float32:
            _the_max_to_calc_threshold__b.to(torch.float16)
            _the_min_to_calc_threshold__b.to(torch.float16)
            pass
        if _log and _needs_log__before_loop:
            _log["before_loop"].append(f"_the_max_to_calc_threshold__b init to:{_the_max_to_calc_threshold__b}")
            _log["before_loop"].append(f"_the_min_to_calc_threshold__b init to:{_the_min_to_calc_threshold__b}")
            pass
        
        #all the zero init.
        _guess_threshold__b = torch.zeros_like(if_finished__b,dtype=_the_max_to_calc_threshold__b.dtype)#.reshape([-1,1])#11111111111111
        _if__guess_too_big___b = torch.zeros_like(if_finished__b)
        _if__guess_too_small___b = torch.zeros_like(if_finished__b)
        _input_gt_guess__count__b = torch.zeros_like(if_finished__b, dtype=int_dtype)
        
        RESULT__if__input_gt_guess__b_i = torch.zeros_like(input__b_i, dtype=torch.bool)
        old_unqualified_RESULT__if__input_gt_guess__b_i = torch.zeros_like(input__b_i,dtype=torch.bool)
        
        _if__unchanged__b = torch.zeros_like(if_finished__b, dtype=torch.bool)
        repeating__b = torch.zeros_like(_the_max_to_calc_threshold__b, dtype=torch.int8)#.squeeze_()#11111111111111
        
        # now is this one: if_finished__b
        # it was init_ed_the_flag_result
        if _log and _needs_log__before_loop:
            _log["before_loop"].append("vvvv   below are all the init to zero   vvvv")
            
            _log["before_loop"].append(f"_guess_threshold__b init to:{_guess_threshold__b}")
            _log["before_loop"].append(f"_if__guess_too_big___b init to:{_if__guess_too_big___b}")
            _log["before_loop"].append(f"_if__guess_too_small___b init to:{_if__guess_too_small___b}")
            _log["before_loop"].append(f"_input_gt_guess__count__b init to:{_input_gt_guess__count__b}")
            
            _log["before_loop"].append(f"RESULT__if__input_gt_guess__b_i init to:{RESULT__if__input_gt_guess__b_i}")
            _log["before_loop"].append(f"old_unqualified_RESULT__b_i init to:{old_unqualified_RESULT__if__input_gt_guess__b_i}")
            
            _log["before_loop"].append(f"_if__unchanged__b init to:{_if__unchanged__b}")
            _log["before_loop"].append(f"repeating__b init to:{repeating__b}")
            pass
        
        #before while
        _needs_log__loop_count = _cpu and (_needs_log__basic_of_loop or _needs_log__binary_search_in_loop or \
                                    _needs_log__error_ratio_in_loop)
        if _needs_log__loop_count:
            if _log is None:
                _log = {}
                pass
        if _cpu and (_log is not None) and _needs_log__loop_count:
            loop_count = 0
            _log["in_loop"] = []
            pass
        while True:
            if _log:
                if _needs_log__loop_count:
                    _log["in_loop"].append(f"----  loop {loop_count}  ----")
                    pass
                if _needs_log__basic_of_loop:
                    _log["in_loop"].append(f"if_finished__b:{if_finished__b}")
                    pass
                pass
            #similar to binary search
            _guess_threshold__b[~if_finished__b] = (_the_max_to_calc_threshold__b[~if_finished__b]+_the_min_to_calc_threshold__b[~if_finished__b])/2.#maybe optimizable.
            if _log and _needs_log__binary_search_in_loop:
                _log["in_loop"].append(f"_guess_threshold:{_guess_threshold__b}")
                pass
            #the real comparison
            RESULT__if__input_gt_guess__b_i[~if_finished__b] = input__b_i[~if_finished__b].gt \
                                                (_guess_threshold__b[~if_finished__b].reshape([-1,1]).expand([-1,_element_per_batch__s]))
            #if guessed too big, then, less true
            _input_gt_guess__count__b[~if_finished__b] = RESULT__if__input_gt_guess__b_i[~if_finished__b].sum(dim=1, dtype=int_dtype)
            #_guess_count = flag_result.to(int_dtype).sum(dim=1)
            if _log and _needs_log__binary_search_in_loop:
                _log["in_loop"].append(f"RESULT_input_gt_guess__b_i___mask_if_finish:{RESULT__if__input_gt_guess__b_i}")
                _log["in_loop"].append(f"_input_gt_guess__count__b:{_input_gt_guess__count__b}")
                pass
            
            
            # #flag_gt old code
            # _if__guess_not_too_big___b = torch.zeros_like(if_finished__b)
            # _if__guess_not_too_big___b[~if_finished__b] = _guess_count__b[~if_finished__b].le(at_most_this_amount__b[~if_finished__b])
            # # ^^^ true is good. ^^^
            # _the_min_to_calc_threshold__b[~_if__guess_not_too_big___b] = _guess_threshold[~_if__guess_not_too_big___b]
            
            #flag_gt
            _if__guess_too_big___b[~if_finished__b] = _input_gt_guess__count__b[~if_finished__b].lt(at_least_this_amount__b[~if_finished__b])
            # ^^^ true is bad. ^^^
            if _log and _needs_log__binary_search_in_loop:
                _log["in_loop"].append(f"_if__guess_too_big___b(true is bad):{_if__guess_too_big___b}")
                _log["in_loop"].append(f"_the_max_to_calc_threshold__b, from:{_the_max_to_calc_threshold__b}")
                pass
            _the_max_to_calc_threshold__b[_if__guess_too_big___b] = _guess_threshold__b[_if__guess_too_big___b]
            if _log and _needs_log__binary_search_in_loop:
                _log["in_loop"].append(f"{_log["in_loop"].pop()}, to:{_the_max_to_calc_threshold__b}")
                pass
            
            
            # #flag_lt old code
            # _if__guess_not_too_small___b = torch.zeros_like(if_finished__b)
            # _if__guess_not_too_small___b[~if_finished__b] = _guess_count__b[~if_finished__b].ge(at_least_this_amount__b[~if_finished__b])
            # # ^^^ true is good. ^^^
            # _the_max_to_calc_threshold__b[~_if__guess_not_too_small___b] = _guess_threshold[~_if__guess_not_too_small___b]
            
            #flag_lt
            _if__guess_too_small___b[~if_finished__b] = _input_gt_guess__count__b[~if_finished__b].gt(at_most_this_amount__b[~if_finished__b])
            # ^^^ true is bad. ^^^
            if _log and _needs_log__binary_search_in_loop:
                _log["in_loop"].append(f"_if__guess_too_small___b(true is bad):{_if__guess_too_small___b}")
                _log["in_loop"].append(f"_the_min_to_calc_threshold__b:{_the_min_to_calc_threshold__b}")
                pass
            _the_min_to_calc_threshold__b[_if__guess_too_small___b] = _guess_threshold__b[_if__guess_too_small___b]
            if _log and _needs_log__binary_search_in_loop:
                _log["in_loop"].append(f"{_log["in_loop"].pop()}, to:{_the_min_to_calc_threshold__b}")
                pass
            
            _flag__not_too_loose__and__not_too_tight___b_1 = (~_if__guess_too_big___b).logical_and(~_if__guess_too_small___b)
            #           ^^^ true is good. ^^^                   ^^^ true is bad. ^^^                  ^^^ true is bad. ^^^  
            if _log: 
                if _needs_log__binary_search_in_loop:
                    _log["in_loop"].append(f"_flag__not_too_loose__and__not_too_tight___b_1(true is good):{_flag__not_too_loose__and__not_too_tight___b_1}")
                    pass
                if _needs_log__basic_of_loop:
                    _log["in_loop"].append(f"if_finished__b, from:{if_finished__b}")
                    pass
                pass
            if_finished__b.logical_or_(_flag__not_too_loose__and__not_too_tight___b_1)
            if _log and _needs_log__basic_of_loop:
                _log["in_loop"].append(f"{_log["in_loop"].pop()}, to:{if_finished__b}")
                pass
            
            if epsilon__for_binary_search is not None:
                _flag_less_than_epsilon = (_the_max_to_calc_threshold__b-_the_min_to_calc_threshold__b).lt(epsilon__for_binary_search__s)
                if _log and _needs_log__binary_search_in_loop:
                    _log["in_loop"].append(f"epsilon__for_binary_search__s:{epsilon__for_binary_search__s}")
                    _log["in_loop"].append(f"[bc epsilon__for_binary_search__s], _flag_less_than_epsilon:{_flag_less_than_epsilon \
                                                    }, and it makes if_finished__b from:{if_finished__b}")
                    pass
                if_finished__b.logical_or_(_flag_less_than_epsilon)
                if _log and _needs_log__binary_search_in_loop:
                    _log["in_loop"].append(f"{_log["in_loop"].pop()}, to:{if_finished__b}")
                    pass
                pass#if epsilon
            
            # this is the only [return] timing.
            if if_finished__b.all():
                if _log:
                    _log["in_loop"].append(f"[return]")
                    pass
                if bottom:
                    if _log:
                        _log["in_loop"].append(f"{_log["in_loop"].pop()}[bc it's bottom=true, returns the reversed result]")
                        pass
                    RESULT__if__input_gt_guess__b_i.logical_not_()
                    pass
                return RESULT__if__input_gt_guess__b_i, _log
                pass #if if_finished__b.all():
            
            
            #if the new result[b,i] unchanged?
            _if__unchanged__b[~if_finished__b] = old_unqualified_RESULT__if__input_gt_guess__b_i[~if_finished__b].eq( \
                                                                RESULT__if__input_gt_guess__b_i[~if_finished__b]).all(dim=1)
            # ^^^ true is bad. ^^^
            
            if _log and _needs_log__error_ratio_in_loop:
                _log["in_loop"].append("-- the second return condition --")
                _log["in_loop"].append(f"_if__unchanged__b:{_if__unchanged__b}")
                _log["in_loop"].append(f"repeating__b, from:{repeating__b}")
                pass
            repeating__b[_if__unchanged__b] = repeating__b[_if__unchanged__b].add(1)
            if _log and _needs_log__error_ratio_in_loop:
                assert _log
                _log["in_loop"].append(f"{_log["in_loop"].pop()}, to:{repeating__b}")
                pass
            
            
            #if 
            _if__repeated_enough__b = repeating__b.ge(careful_level__s)
            if _log and _needs_log__error_ratio_in_loop:
                _log["in_loop"].append(f"_if__repeated_enough__b:{_if__repeated_enough__b}")
                _log["in_loop"].append(f"repeating__b, from:{repeating__b}")
                pass
            repeating__b[_if__repeated_enough__b] = 0
            if _log and _needs_log__error_ratio_in_loop:
                _log["in_loop"].append(f"{_log["in_loop"].pop()}, to:{repeating__b}")
                _log["in_loop"].append(f"error_of_ratio__b, from:{error_of_ratio__b}")
                pass
            #update the finishing flags.
            error_of_ratio__b[_if__repeated_enough__b] = error_of_ratio__b[_if__repeated_enough__b].mul(2.)#this 2. is not tested.
            #maybe wrong??? is it updated?
            if _log and _needs_log__error_ratio_in_loop:
                _log["in_loop"].append(f"{_log["in_loop"].pop()}, to:{error_of_ratio__b}")
                pass
            
            
            
            if _log:
                if _needs_log__error_ratio_in_loop:
                    _log["in_loop"].append(f"_if__repeated_enough__b:{_if__repeated_enough__b}")
                    pass
                if _needs_log__basic_of_loop:
                    _log["in_loop"].append(f"at_least_this_amount__b, from:{at_least_this_amount__b}")
                    _log["in_loop"].append(f"at_most_this_amount__b, from:{at_most_this_amount__b}")
                    pass
                pass
            #ratio+-error, this segment appears twice in this function.
            #[1]+[] is []. So this is safe.
            
            at_least_this_amount__b[_if__repeated_enough__b] = ((_element_per_batch__s-2)*(top_ratio__s - \
                error_of_ratio__b[_if__repeated_enough__b])+1.4999).to(int_dtype)
            at_most_this_amount__b[_if__repeated_enough__b] =  ((_element_per_batch__s-2)*(top_ratio__s + \
                error_of_ratio__b[_if__repeated_enough__b])+1.5001).to(int_dtype)
            # no detect for return here. reason:
            # even if this range-like can mean a range covering all the range, bc I believe it unlikely to happen.
            # I decide to delay the return to the next round.
            if _log and _needs_log__basic_of_loop: 
                _temp_str_at_most_this_amount__b_from = _log["in_loop"].pop()
                _log["in_loop"].append(f"{_log["in_loop"].pop()}, to:{at_least_this_amount__b}")
                _log["in_loop"].append(f"{_temp_str_at_most_this_amount__b_from}, to:{at_most_this_amount__b}")
                pass
            
            #tail
            if _log and _needs_log__binary_search_in_loop:
                _log["in_loop"].append(f"RESULT__b_i, from:{old_unqualified_RESULT__if__input_gt_guess__b_i}, to:{RESULT__if__input_gt_guess__b_i}")
                pass
            old_unqualified_RESULT__if__input_gt_guess__b_i = RESULT__if__input_gt_guess__b_i
            if _log and _needs_log__loop_count:
                _log["in_loop"].append(f"loop {loop_count} ends.")
                loop_count += 1
                pass
            pass#while true
        
        pass#  no_grad
    pass# end of function
if "performance test    slow" and __DEBUG_ME__() and False:
    "result"
    "my version is basically only about the amount of data. torch version is better unless it's a lot data and on cpu. "
    "so, although I wrote my version, but it's still faster to move the data to gpu and do torch version."
    "ok, this is cool."
    # cpu: my:0.001839, torch:0.000041   [  10,  100]   gpu: my:0.013041, torch:0.000045   torch move to gpu: my:0.000063   [  10,  100]
    # cpu: my:0.001658, torch:0.000178   [  10,  330]   gpu: my:0.015410, torch:0.000037   torch move to gpu: my:0.000064   [  10,  330]
    # cpu: my:0.001906, torch:0.000672   [  10, 1000]   gpu: my:0.010878, torch:0.000035   torch move to gpu: my:0.000064   [  10, 1000]
    # cpu: my:0.002109, torch:0.001553   [  10, 3300]   gpu: my:0.011062, torch:0.000042   torch move to gpu: my:0.000094   [  10, 3300]
    # cpu: my:0.002413, torch:0.003691   [  10,10000]   gpu: my:0.011196, torch:0.000230   torch move to gpu: my:0.000253   [  10,10000]
    # cpu: my:0.002128, torch:0.000131   [  33,  100]   gpu: my:0.013590, torch:0.000046   torch move to gpu: my:0.000073   [  33,  100]
    # cpu: my:0.002270, torch:0.000586   [  33,  330]   gpu: my:0.014319, torch:0.000048   torch move to gpu: my:0.000064   [  33,  330]
    # cpu: my:0.002400, torch:0.001306   [  33, 1000]   gpu: my:0.010810, torch:0.000043   torch move to gpu: my:0.000094   [  33, 1000]
    # cpu: my:0.002779, torch:0.003278   [  33, 3300]   gpu: my:0.010969, torch:0.000091   torch move to gpu: my:0.000163   [  33, 3300]
    # cpu: my:0.003225, torch:0.004388   [  33,10000]   gpu: my:0.013599, torch:0.000470   torch move to gpu: my:0.000599   [  33,10000]
    # cpu: my:0.002397, torch:0.000412   [ 100,  100]   gpu: my:0.013237, torch:0.000046   torch move to gpu: my:0.000070   [ 100,  100]
    # cpu: my:0.002473, torch:0.001105   [ 100,  330]   gpu: my:0.014954, torch:0.000047   torch move to gpu: my:0.000100   [ 100,  330]
    # cpu: my:0.002923, torch:0.002156   [ 100, 1000]   gpu: my:0.015217, torch:0.000062   torch move to gpu: my:0.000132   [ 100, 1000]
    # cpu: my:0.003317, torch:0.003772   [ 100, 3300]   gpu: my:0.011056, torch:0.000224   torch move to gpu: my:0.000372   [ 100, 3300]
    # cpu: my:0.006159, torch:0.013583   [ 100,10000]   gpu: my:0.011261, torch:0.001141   torch move to gpu: my:0.001530   [ 100,10000]
    # cpu: my:0.003091, torch:0.000771   [ 330,  100]   gpu: my:0.019449, torch:0.000058   torch move to gpu: my:0.000093   [ 330,  100]
    # cpu: my:0.003503, torch:0.001913   [ 330,  330]   gpu: my:0.014683, torch:0.000103   torch move to gpu: my:0.000179   [ 330,  330]
    # cpu: my:0.003942, torch:0.003066   [ 330, 1000]   gpu: my:0.013820, torch:0.000172   torch move to gpu: my:0.000320   [ 330, 1000]
    # cpu: my:0.006749, torch:0.012893   [ 330, 3300]   gpu: my:0.011935, torch:0.000608   torch move to gpu: my:0.001016   [ 330, 3300]
    # cpu: my:0.018486, torch:0.041036   [ 330,10000]   gpu: my:0.013873, torch:0.004045   torch move to gpu: my:0.005211   [ 330,10000]
    # cpu: my:0.003851, torch:0.001376   [1000,  100]   gpu: my:0.019149, torch:0.000077   torch move to gpu: my:0.000159   [1000,  100]
    # cpu: my:0.004946, torch:0.002477   [1000,  330]   gpu: my:0.014508, torch:0.000281   torch move to gpu: my:0.000426   [1000,  330]
    # cpu: my:0.007018, torch:0.008683   [1000, 1000]   gpu: my:0.015103, torch:0.000425   torch move to gpu: my:0.000792   [1000, 1000]
    # cpu: my:0.019464, torch:0.036045   [1000, 3300]   gpu: my:0.017305, torch:0.001718   torch move to gpu: my:0.002898   [1000, 3300]
    # cpu: my:0.060130, torch:0.116552   [1000,10000]   gpu: my:0.021209, torch:0.010704   torch move to gpu: my:0.015695   [1000,10000]
    
    def ____test____is_my_version_faster():
        from pytorch_yagaodirac_v2.timeit_yagaodirac import timeit
        for batch in [10,33,100,330,1000]:
            for dim in [100,330,1000,3300,10000]:
                size = torch.Size([batch, dim])
                if batch*dim<=1_000_000:
                    time_at_most = 1.
                    pass
                else:
                    time_at_most = 4.
                    pass
                
                # vec = torch.rand(size=size)
                # def my_version():
                #     a = get_mask_of_top_element__rough(vec)
                #     pass
                # my_version_time = timeit(my_version, time_at_most=time_at_most)
                
                # vec = torch.rand(size=size)
                # def torch_version():
                #     b = vec.sort().values[:,:9000]
                #     pass
                # torch_version_time = timeit(torch_version, time_at_most=time_at_most)
                
                # vec_gpu = torch.rand(size=size,device='cuda')
                # def my_gpu_version():
                #     a = get_mask_of_top_element__rough(vec_gpu)
                #     pass
                # my_gpu_version_time = timeit(my_gpu_version, time_at_most=time_at_most)
                
                # vec_gpu = torch.rand(size=size,device='cuda')
                # def torch_gpu_version():
                #     b = vec_gpu.sort().values[:,:9000]
                #     pass
                # torch_gpu_version_time = timeit(torch_gpu_version, time_at_most=time_at_most)
                
                vec = torch.rand(size=size)
                def torch_move_to_gpu_version():
                    b = vec.to('cuda').sort().values[:,:9000]
                    pass
                torch_move_to_gpu_version_time = timeit(torch_move_to_gpu_version, time_at_most=time_at_most)
                
                #print(f"cpu: my:{my_version_time[0]:.6f}, torch:{torch_version_time[0]:.6f}   [{batch:4},{dim:5
                #   }]   gpu: my:{my_gpu_version_time[0]:.6f}, torch:{torch_gpu_version_time[0]:.6f}")
                print(f"cpu: my:{torch_move_to_gpu_version_time[0]:.6f}   [{batch:4},{dim:5}]")
                pass
            pass
        return
    ____test____is_my_version_faster()
    pass
if "test" and __DEBUG_ME__() and False:
    
    if "some real case,      batch>1      " and False:
        _input = torch.ones(size=[6,11])
        _input[0] = _input[0] + torch.randn(size=[1,11])*0.001
        _input[1] = _input[1]*-1 + torch.randn(size=[1,11])*0.001
        _input[2,0] = 0.
        _input[3,0] = 1e-10
        _input[4,0] = 1e-21
        _input[4,1] = 1e-10
        _input[5,0] = 1e-10
        _input[5,1] = 1e10
        
        log_of_input = _input.abs().log10()
        no_nan_log = log_of_input.nan_to_num(-999999.,posinf=-999999.,neginf=-999999.) 
        
        _result_tuple = get_mask_of_top_element__rough(no_nan_log, \
                            _needs_log__before_loop = True, _needs_log__basic_of_loop = True, \
                        _needs_log__binary_search_in_loop = True, _needs_log__error_ratio_in_loop = True)
        _log = _result_tuple[1]
        #a = _log["before_loop"]
        #b = _log["in_loop"]
        pass
    
    
    # torch.topk is not what I need.
    # a = torch.topk(torch.tensor([1,2,3,4,5]),3, sorted=False)
    # b = torch.topk(torch.tensor([1,2,3,4,5]),3, sorted=True)
    
    if "to test the formula for bounds" and False:
        n = 5
        top_ratio_list:list[float] = []
        for ii in range(1,10):
            top_ratio_list.append(ii*0.1)
            pass
        error_of_ratio = 0.1
        _floor_offset = 1.
        
        for top_ratio in top_ratio_list:
            lower_bound = (n-2)*(top_ratio - error_of_ratio)+_floor_offset
            upper_bound = (n-2)*(top_ratio + error_of_ratio)+_floor_offset
            # print(f"{top_ratio:.2f}, {lower_bound:.2f}/{upper_bound:.2f}")
            # from 1. to 4.
            # but in the real case, the offset is around 1.5, bc it's truncated into integer later.
            pass
        
        n = 5
        top_ratio_list = []
        for ii in range(1,30):
            top_ratio_list.append(ii*0.01)
            pass
        error_of_ratio = 0.1
        _floor_offset = 1.5
        
        for top_ratio in top_ratio_list:
            lower_bound = (n-2)*(top_ratio - error_of_ratio)+_floor_offset
            upper_bound = (n-2)*(top_ratio + error_of_ratio)+_floor_offset
            #print(f"{top_ratio:.2f}, {lower_bound:.2f}/{upper_bound:.2f}")
            pass
        
        for top_ratio in [0.06,0.07, 0.39,0.4, 0.73,0.74]:
            lower_bound = (n-2)*(top_ratio - error_of_ratio)+_floor_offset
            upper_bound = (n-2)*(top_ratio + error_of_ratio)+_floor_offset
            #print(f"{top_ratio:.2f}, {lower_bound:.2f}/{upper_bound:.2f}")
            pass
        
        #but this one looks symmetry.
        n = 10
        top_ratio_list = []
        for ii in range(1,100):
            top_ratio_list.append(ii*0.01)
            pass
        error_of_ratio = 0.1
        _floor_offset = 1.5
        
        for top_ratio in top_ratio_list:
            lower_bound = (n-2)*(top_ratio - error_of_ratio)+_floor_offset
            upper_bound = (n-2)*(top_ratio + error_of_ratio)+_floor_offset
            #print(f"{top_ratio:.2f}, {lower_bound:.2f}/{upper_bound:.2f}")
        pass
    
    
    
    
    #a1 = get_mask_of_top_element__rough(torch.tensor([[1.,2,3,4,5]]),top_ratio=0.01, _debug_needs_log = True)
    #a2 = get_mask_of_top_element__rough(torch.tensor([[1.,2,3,4,5]]),top_ratio=0.33, _debug_needs_log = True)
    #a3 = get_mask_of_top_element__rough(torch.tensor([[1.,2,3,4,5]]),top_ratio=0.67, _debug_needs_log = True)
    #a4 = get_mask_of_top_element__rough(torch.tensor([[1.,2,3,4,5]]),top_ratio=0.99, _debug_needs_log = True)
    assert get_mask_of_top_element__rough(torch.tensor([[1.,2,3,4,5]]),top_ratio=0.01)[0].eq(torch.tensor([False,False,False,False,True])).all()
    assert get_mask_of_top_element__rough(torch.tensor([[1.,2,3,4,5]]),top_ratio=0.33)[0].eq(torch.tensor([False,False,False,True,True])).all()
    assert get_mask_of_top_element__rough(torch.tensor([[1.,2,3,4,5]]),top_ratio=0.67)[0].eq(torch.tensor([False,False,True,True,True])).all()
    assert get_mask_of_top_element__rough(torch.tensor([[1.,2,3,4,5]]),top_ratio=0.99)[0].eq(torch.tensor([False,True,True,True,True])).all()
    
    assert get_mask_of_top_element__rough(torch.tensor([[5.,2,3,4,1]]),top_ratio=0.01)[0].eq(torch.tensor([True,False,False,False,False])).all()
    assert get_mask_of_top_element__rough(torch.tensor([[5.,2,3,4,1]]),top_ratio=0.33)[0].eq(torch.tensor([True,False,False,True,False])).all()
    

    assert get_mask_of_top_element__rough(torch.tensor([[1.,2,3,4,5]]),top_ratio=0.01,bottom=True)[0].eq(torch.tensor([True,False,False,False,False])).all()
    assert get_mask_of_top_element__rough(torch.tensor([[1.,2,3,4,5]]),top_ratio=0.33,bottom=True)[0].eq(torch.tensor([True,True,False,False,False])).all()
    assert get_mask_of_top_element__rough(torch.tensor([[1.,2,3,4,5]]),top_ratio=0.67,bottom=True)[0].eq(torch.tensor([True,True,True,False,False])).all()
    assert get_mask_of_top_element__rough(torch.tensor([[1.,2,3,4,5]]),top_ratio=0.99,bottom=True)[0].eq(torch.tensor([True,True,True,True,False])).all()
    
    assert get_mask_of_top_element__rough(torch.tensor([[1.,2,3]]),top_ratio=0.01)[0].eq(torch.tensor([False,False,True])).all()
    assert get_mask_of_top_element__rough(torch.tensor([[1.,2,3]]),top_ratio=0.99)[0].eq(torch.tensor([False,True,True])).all()
    
    assert get_mask_of_top_element__rough(torch.tensor([[5.,2,3]]),top_ratio=0.01)[0].eq(torch.tensor([True,False,False])).all()
    assert get_mask_of_top_element__rough(torch.tensor([[5.,2,3]]),top_ratio=0.99)[0].eq(torch.tensor([True,False,True])).all()
    
    assert get_mask_of_top_element__rough(torch.tensor([[1.,2,3]]),top_ratio=0.01,bottom=True)[0].eq(torch.tensor([True,False,False])).all()
    assert get_mask_of_top_element__rough(torch.tensor([[1.,2,3]]),top_ratio=0.99,bottom=True)[0].eq(torch.tensor([True,True,False])).all()
    
        
    _shift_by_list = [          6,          7,         14,         15,]
    _int_dtype_list = [ torch.int8, torch.int16, torch.int16,torch.int32]
    for ii in range(_shift_by_list.__len__()):
        _shift_by = _shift_by_list[ii]
        _1_left_shift_by_shift_by_ = 1<<_shift_by
        for dtype in [torch.float16, torch.float32, torch.float64]:
            _input = torch.rand(size=[1,_1_left_shift_by_shift_by_],dtype=dtype)
            _result_tuple__tensor__list = get_mask_of_top_element__rough(_input,top_ratio=0.9, _needs_log__before_loop = True, \
                                                                                            _needs_log__basic_of_loop = True)
            _the_sum = _result_tuple__tensor__list[0].sum().item()
            assert _the_sum>(_1_left_shift_by_shift_by_)*0.8
            assert _the_sum<(_1_left_shift_by_shift_by_)
            
            _log = _result_tuple__tensor__list[1]
            assert _log
            assert _log["before_loop"][1]
            assert _log["before_loop"][1] == f"int type:{_int_dtype_list[ii]}"
            assert _log["in_loop"].__len__() >1
            
        pass

    # epsilon
    # epsilon helps when all elements are equal or nearly equal.
    # When the ratio doesn't exist, it's also helpful. At least the function returns.
    _result_tuple__tensor__list = get_mask_of_top_element__rough(torch.tensor([[1.,1,1,1,1]]),top_ratio=0.5, bottom=True, epsilon__for_binary_search=0.001, \
                    _needs_log__before_loop = True,             _needs_log__basic_of_loop = True, \
                    _needs_log__binary_search_in_loop = True,   _needs_log__error_ratio_in_loop = True, )
    _log = _result_tuple__tensor__list[1]
    assert _log
    assert _log["in_loop"][-2]  == f"[bc epsilon__for_binary_search__s], _flag_less_than_epsilon:{torch.tensor([True])\
                                        }, and it makes if_finished__b from:{torch.tensor([False])}, to:{torch.tensor([True])}"
    
    _input = torch.zeros(size=[1,30])
    _input[0,0] = -1.
    _input[0,-1] = 1.
    _result_tuple__tensor__list = get_mask_of_top_element__rough(_input,top_ratio=0.5, epsilon__for_binary_search=0.01, \
                    _needs_log__binary_search_in_loop = True,   _needs_log__error_ratio_in_loop = True, )
    _log = _result_tuple__tensor__list[1]
    assert _log
    assert _log["in_loop"][-2]  == f"[bc epsilon__for_binary_search__s], _flag_less_than_epsilon:{torch.tensor([True])\
                                        }, and it makes if_finished__b from:{torch.tensor([False])}, to:{torch.tensor([True])}"
    assert isinstance(_log["in_loop"], list)
    _temp_log_len_for_epsilon_0_01 = _log["in_loop"].__len__()
    
    _result_tuple__tensor__list = get_mask_of_top_element__rough(_input,top_ratio=0.5, epsilon__for_binary_search=0.1, \
                    _needs_log__binary_search_in_loop = True,   _needs_log__error_ratio_in_loop = True, )
    _log = _result_tuple__tensor__list[1]
    assert _log
    assert _log["in_loop"][-2]  == f"[bc epsilon__for_binary_search__s], _flag_less_than_epsilon:{torch.tensor([True])\
                                        }, and it makes if_finished__b from:{torch.tensor([False])}, to:{torch.tensor([True])}"
    assert isinstance(_log["in_loop"], list)
    _temp_log_len_for_epsilon_0_1 = _log["in_loop"].__len__()
    
    _result_tuple__tensor__list = get_mask_of_top_element__rough(_input,top_ratio=0.5, epsilon__for_binary_search=1., \
                    _needs_log__binary_search_in_loop = True,   _needs_log__error_ratio_in_loop = True, )
    _log = _result_tuple__tensor__list[1]
    assert _log
    assert _log["in_loop"][-2]  == f"[bc epsilon__for_binary_search__s], _flag_less_than_epsilon:{torch.tensor([True])\
                                        }, and it makes if_finished__b from:{torch.tensor([False])}, to:{torch.tensor([True])}"
    assert isinstance(_log["in_loop"], list)
    _temp_log_len_for_epsilon_1 = _log["in_loop"].__len__()
    
    assert _temp_log_len_for_epsilon_0_01>_temp_log_len_for_epsilon_0_1
    assert _temp_log_len_for_epsilon_0_1>_temp_log_len_for_epsilon_1
    
    
        
        
    #epsilon__for_binary_search as tensor
    _result_tuple__tensor__list = get_mask_of_top_element__rough(torch.tensor([[1.,1,1,1,1]]),top_ratio=0.5, \
                                                                    epsilon__for_binary_search=torch.tensor(0.001), \
                                                                _needs_log__binary_search_in_loop = True,  )
    _log = _result_tuple__tensor__list[1]
    assert _log
    assert _log["in_loop"][-2]  == f"[bc epsilon__for_binary_search__s], _flag_less_than_epsilon:{torch.tensor([True])\
                                            }, and it makes if_finished__b from:{torch.tensor([False])}, to:{torch.tensor([True])}"



    #gpu
    _result_tuple__tensor__list = get_mask_of_top_element__rough(torch.tensor([[1.,2,3]], device='cuda'),top_ratio=0.25)
    assert _result_tuple__tensor__list[0].device.type == "cuda"
    assert _result_tuple__tensor__list[0].eq(torch.tensor([False,False,True ],device='cuda')).all()
    _result_tuple__tensor__list = get_mask_of_top_element__rough(torch.tensor([[1.,2,3]],device='cuda'),top_ratio=0.75)
    assert _result_tuple__tensor__list[0].device.type == "cuda"
    assert _result_tuple__tensor__list[0].eq(torch.tensor([False,True,True ],device='cuda')).all()
    #gpu has no log.
    _result_tuple__tensor__list = get_mask_of_top_element__rough(torch.tensor([[1.,2,3]],device='cuda'), _needs_log__before_loop = True)
    assert _result_tuple__tensor__list[1] is None
    
        
    #error_of_ratio__at_least
    the_linspace = torch.linspace(1.,100.,99).reshape([1,-1])
    _result_tuple__tensor__list = get_mask_of_top_element__rough(the_linspace,top_ratio=0.2, error_of_ratio__at_least=0.01, \
                                                                    _needs_log__basic_of_loop = True)
    _temp_int = _result_tuple__tensor__list[0].sum().item()
    assert _temp_int>20-2 and _temp_int<20+2
    _log = _result_tuple__tensor__list[1]
    assert _log
    _log_len_for__error_ratio_0_01 = _log["in_loop"].__len__()
    
    _result_tuple__tensor__list = get_mask_of_top_element__rough(the_linspace,top_ratio=0.2, error_of_ratio__at_least=0.1, \
                                                                    _needs_log__basic_of_loop = True)
    _temp_int = _result_tuple__tensor__list[0].sum().item()
    assert _temp_int>20-11 and _temp_int<20+11
    _log = _result_tuple__tensor__list[1]
    assert _log
    _log_len_for__error_ratio_0_1 = _log["in_loop"].__len__()
    
    assert _log_len_for__error_ratio_0_01>_log_len_for__error_ratio_0_1
    

    
    # careful_level
    # when the binary search doesn't work stably, this careful_level controls the second way of exit.
    # when the loop repeats too much without any progress, the real error_of_ratio(or maybe something else) is modified to 
    # eventually break the loop. The result maybe rougher, but at least, you get some result.
    _input = torch.zeros(size=[1,100])
    _input[0,0] = 99999
    #step into the function and see how it works.
    _result_tuple__tensor__list = get_mask_of_top_element__rough(_input, top_ratio=0.5, error_of_ratio__at_least=0.0000001, \
                                        careful_level = 1, 
                                            _needs_log__basic_of_loop = True, _needs_log__binary_search_in_loop = True, \
                                            _needs_log__error_ratio_in_loop = True)
    _log = _result_tuple__tensor__list[1]
    assert _log
    _from_list = [0.005,0.005,0.01, 0.02,0.04,0.08,0.16]
    _to_list =   [0.005, 0.01,0.02, 0.04,0.08,0.16,0.32]
    _epsi_list = [0.001,0.001,0.001,0.01,0.01,0.01,0.01]
    
    for ii in range((180-16)//22):
        log_index = ii*22 +16
        _log_item = _log["in_loop"][log_index]
        _pos_0 = _log_item.find("error_of_ratio__b, from:tensor([", 0)
        assert _pos_0 == 0
        _pos_1 = _log_item.find("]), to:", 32)
        _pos_2 = _log_item.find("])", 40)
        #aaaaaa = _log_item[32:_pos_1]
        _number_from = float(_log_item[32:_pos_1])
        _float_equal(_number_from, _from_list[ii], epsilon=_epsi_list[ii])
        #bbbbbb = _log_item[_pos_1+15:_pos_2]
        _number_to = float(_log_item[_pos_1+15:_pos_2])
        _float_equal(_number_to, _to_list[ii], epsilon=_epsi_list[ii])
        pass
        
    
    
    # batch>1
    the_tensor = torch.tensor( [[1.,2,3,4,5],
                                [5.,2,3,4,1]])
    the_result = torch.tensor([[False,False,False,False,True],
                                [True,False,False,False,False]])
    _result_tuple__tensor__list = get_mask_of_top_element__rough(the_tensor, top_ratio=0.01)
    assert _result_tuple__tensor__list[0].eq(the_result).all()
    the_tensor = torch.tensor( [[1.,2,3,4,5],
                                [5.,2,3,4,1]], device='cuda')
    the_result = torch.tensor([[False,False,False,False,True],
                                [True,False,False,False,False]], device='cuda')
    _result_tuple__tensor__list = get_mask_of_top_element__rough(the_tensor, top_ratio=0.01)
    assert _result_tuple__tensor__list[0].eq(the_result).all()
    pass
if "old version of get_mask_of_top_element__rough function" and False:
    def 应该是不用了get_top_ratio如果没改就不要了_上面已经搞定了(input:torch.Tensor, top_ratio = 0.5, error_of_ratio = 0.01, \
                                bottom = False)->torch.Tensor:
        ''' 
        return shape is the same as input. dtype of return is torch.bool.
        
        if shape is too small, this may not work.
        '''
        assert input.shape.__len__()==2
        nelement_per_batch__s = input.shape[1]
        with torch.no_grad():
            #safety first
            _at_least_this_amount__cpu_int = int(nelement_per_batch__s*(top_ratio - error_of_ratio)+0.4999999999999)
            at_least_this_amount__s = torch.tensor(_at_least_this_amount__cpu_int, device=input.device)
            _at_most_this_amount__cpu_int =  int(nelement_per_batch__s*(top_ratio + error_of_ratio)+0.4999999999999)
            at_most_this_amount__s =  torch.tensor(_at_most_this_amount__cpu_int, device=input.device)
            # if at_least_this_amount == at_most_this_amount: xxxxxxxxxxxxxx
            #     at_most_this_amount = at_least_this_amount  +1
            #     pass
            if _at_least_this_amount__cpu_int >= nelement_per_batch__s:
                _temp_tensor = torch.ones_like(input, dtype=torch.bool, device=input.device)
                return _temp_tensor
            if _at_most_this_amount__cpu_int <= 0.:
                _temp_tensor = torch.zeros_like(input, dtype=torch.bool, device=input.device)
                return _temp_tensor
            assert error_of_ratio>=0.
            
            #real job.
            #best dtype for count the amount.
            if nelement_per_batch__s<=(1<<8):
                dtype = torch.uint8
                pass
            elif nelement_per_batch__s<=(1<<16):
                dtype = torch.uint16
                pass
            elif nelement_per_batch__s<=(1<<32):
                dtype = torch.uint32
                pass
            else:
                dtype = torch.uint64
                pass
            # device = input.device
            # param_factory = {"device":device, "dtype":dtype}
            
            #init before loop
            _the_max_threshold__b:torch.Tensor = input.max(dim=1).values.to(torch.float64)
            _the_min_threshold__b:torch.Tensor = input.min(dim=1).values.to(torch.float64)
            #1w 加一个flag
            #1w 加一个flag
            #1w 加一个flag
            #1w 加一个flag
            #1w 加一个flag
            #1w 加一个flag
            #1w 加一个flag
            #1w 加一个flag
            #1w 加一个flag
            #1w 加一个flag
            #1w 加一个flag
            #1w 加一个flag
            #1w 加一个flag
            #1w 加一个flag
            while True:
                #1w 加一个强制退出条件。
                #1w 加一个强制退出条件。
                #1w 加一个强制退出条件。
                #1w 加一个强制退出条件。
                #1w 加一个强制退出条件。
                #1w 加一个强制退出条件。
                #1w 加一个强制退出条件。
                #1w 加一个强制退出条件。
                #1w 加一个强制退出条件。
                #1w 加一个强制退出条件。
                #1w 加一个强制退出条件。
                #1w 加一个强制退出条件。
                #1w 加一个强制退出条件。
                #1w 加一个强制退出条件。
                #1w 加一个强制退出条件。
                #1w 加一个强制退出条件。
                _guess_threshold = (_the_max_threshold__b+_the_min_threshold__b)/2.
                if bottom:
                    flag_result = input.lt(_guess_threshold)
                    _guess_count = flag_result.to(dtype).sum()
                    if _guess_count>at_most_this_amount__s:
                        _the_max_threshold__b = _guess_threshold
                        pass
                    elif _guess_count<at_least_this_amount__s:
                        _the_min_threshold__b = _guess_threshold
                        pass
                    else:
                        return flag_result
                    pass#if bottom:
                else:#top
                    flag_result = input.gt(_guess_threshold)
                    _guess_count = flag_result.to(dtype).sum()
                    if _guess_count>at_most_this_amount__s:
                        _the_min_threshold__b = _guess_threshold
                        pass
                    elif _guess_count<at_least_this_amount__s:
                        _the_max_threshold__b = _guess_threshold
                        pass
                    else:
                        return flag_result
                    pass#top
                    
                pass#while
            pass#  no_grad
        pass# end of function
    pass# old code.        

def _raw_log10_avg_safe__with_batch(input:torch.Tensor, top_ratio = 0.9, careful_level:int|None=None)->torch.Tensor:
    '''check out the log10 measurement.py to see how this measurement function helps.
    
    This is the old version. Result is slightly different from the new version.
    
    If you are not using torch, or a version much newer than 2.9, maybe this version is better?
    This version output the measurement for each batch. If the entire thing is in 1 batch, 
    use the new version(log10_avg_safe), it's always faster.
    
    I found the new way to accelerate this function is always gpu+torch.sort
    So, if a data is on cpu, so move it to gpu and sort it and get the amount, avg, return.
    Yeah, the old my version retired. No need for a mask anymore.
    
    
    old docs below. 
    
    Calcs the average of log10 of abs of input. 
    
    The least log intermediate results are ignored. Because if a number is very close to 0, 
    the log10 of it is very negative, and any noise on such elements will introduce a bit 
    noise into the final result. So they are ignored.

    Inside this function, it calls get_mask_of_top_element__rough to help filter the
    bad intermediate results. That function also helps in a lot other cases.
    
    This function is mainly designed to help extract info from tensors, and to help
    measure some aspects in neural network training dynamics.
    
    If you don't like the shape convention and you know what you are doing, fell free
    to modify this function and the inner get_mask_of_top_element__rough function.
    '''
    #assert input.shape.__len__() == 2, "my convention, shape is [batch, anything]"
    assert input.shape.__len__() <=2
    
    if input.shape.__len__() == 1:
        ori_shape_is_1d = True
        input = input.reshape([1,-1])
        pass
    else:
        ori_shape_is_1d = False
        pass
    
    with torch.no_grad():
        log_of_input = input.abs().log10()
        #safety
        no_nan_log = log_of_input.nan_to_num(-999999.,posinf=-999999.,neginf=-999999.) 
        #safe_log is safe.
        useful_flag:torch.Tensor = get_mask_of_top_element__rough(no_nan_log, top_ratio = top_ratio,\
                                                                    careful_level=careful_level)[0]
        _masked_tensor = torch.masked.masked_tensor(no_nan_log,useful_flag)
        _masked_mean = _masked_tensor.mean(dim=1)
        assert hasattr(_masked_mean, "_masked_data")
        assert hasattr(_masked_mean, "_masked_mask")
        _masked_mean_data:torch.Tensor = _masked_mean._masked_data
        _masked_mean_data[_masked_mean._masked_mask.logical_not()] = torch.nan
        
        if ori_shape_is_1d:
            input = input.reshape([-1])
            pass
        
        return _masked_mean_data
    pass#end of function
if "top ratio" and __DEBUG_ME__() and False:
    "result. diff from 0.99 to 0.999 is 0.010, "
    "                  0.9  to 0.99  is 0.099, "
    "                  0.6  to 0.9   is 0.184,   this holds across all the scale_factor from 1e-3 to 1e3."
    "this result is only for the randn. It's a bit different for rand."
    #prin(-3.268+3.258,-3.258+3.159,-3.159+2.975,)
    #prin(-0.268+0.258,-0.258+0.159,-0.159- 0.025,)
    #prin(2.732-2.742,2.742-2.841,2.841-3.025,)
    
    
    #randn
    # scale_factor= 0.001, top_ratio=0.999, avg=-3.268, std=0.0078
    # scale_factor= 0.001, top_ratio=0.990, avg=-3.258, std=0.0115
    # scale_factor= 0.001, top_ratio=0.980, avg=-3.243, std=0.0086
    # scale_factor= 0.001, top_ratio=0.970, avg=-3.229, std=0.0081
    # scale_factor= 0.001, top_ratio=0.960, avg=-3.216, std=0.0076
    # scale_factor= 0.001, top_ratio=0.900, avg=-3.159, std=0.0094
    # scale_factor= 0.001, top_ratio=0.600, avg=-2.975, std=0.0062
    # scale_factor=   1.0, top_ratio=0.999, avg=-0.268, std=0.0079
    # scale_factor=   1.0, top_ratio=0.990, avg=-0.258, std=0.0113
    # scale_factor=   1.0, top_ratio=0.980, avg=-0.243, std=0.0086
    # scale_factor=   1.0, top_ratio=0.970, avg=-0.229, std=0.0082
    # scale_factor=   1.0, top_ratio=0.960, avg=-0.216, std=0.0076
    # scale_factor=   1.0, top_ratio=0.900, avg=-0.159, std=0.0098
    # scale_factor=   1.0, top_ratio=0.600, avg= 0.025, std=0.0062
    # scale_factor=1000.0, top_ratio=0.999, avg= 2.732, std=0.0079
    # scale_factor=1000.0, top_ratio=0.990, avg= 2.742, std=0.0113
    # scale_factor=1000.0, top_ratio=0.980, avg= 2.757, std=0.0086
    # scale_factor=1000.0, top_ratio=0.970, avg= 2.771, std=0.0082
    # scale_factor=1000.0, top_ratio=0.960, avg= 2.784, std=0.0077
    # scale_factor=1000.0, top_ratio=0.900, avg= 2.841, std=0.0098
    # scale_factor=1000.0, top_ratio=0.600, avg= 3.025, std=0.0062
    
    #rand
    # scale_factor= 0.001, top_ratio=0.999, avg=-3.423, std=0.0074
    # scale_factor= 0.001, top_ratio=0.990, avg=-3.417, std=0.0083
    # scale_factor= 0.001, top_ratio=0.980, avg=-3.402, std=0.0087
    # scale_factor= 0.001, top_ratio=0.970, avg=-3.388, std=0.0075
    # scale_factor= 0.001, top_ratio=0.960, avg=-3.377, std=0.0068
    # scale_factor= 0.001, top_ratio=0.900, avg=-3.323, std=0.0080
    # scale_factor= 0.001, top_ratio=0.600, avg=-3.169, std=0.0048
    # scale_factor=   1.0, top_ratio=0.999, avg=-0.423, std=0.0074
    # scale_factor=   1.0, top_ratio=0.990, avg=-0.417, std=0.0083
    # scale_factor=   1.0, top_ratio=0.980, avg=-0.402, std=0.0087
    # scale_factor=   1.0, top_ratio=0.970, avg=-0.388, std=0.0074
    # scale_factor=   1.0, top_ratio=0.960, avg=-0.377, std=0.0070
    # scale_factor=   1.0, top_ratio=0.900, avg=-0.323, std=0.0079
    # scale_factor=   1.0, top_ratio=0.600, avg=-0.169, std=0.0051
    # scale_factor=1000.0, top_ratio=0.999, avg= 2.577, std=0.0074
    # scale_factor=1000.0, top_ratio=0.990, avg= 2.583, std=0.0083
    # scale_factor=1000.0, top_ratio=0.980, avg= 2.598, std=0.0087
    # scale_factor=1000.0, top_ratio=0.970, avg= 2.612, std=0.0075
    # scale_factor=1000.0, top_ratio=0.960, avg= 2.623, std=0.0069
    # scale_factor=1000.0, top_ratio=0.900, avg= 2.677, std=0.0081
    # scale_factor=1000.0, top_ratio=0.600, avg= 2.831, std=0.0045
    
    def ____test____top_ratio_scan_____raw_log10_avg_safe__with_batch():
        for scale_factor in [1e-3,1.,1e3]:
            #for top_ratio in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2 ,0.1]:
            #for top_ratio in [0.999, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.9, 0.8, 0.7, 0.6, 0.5]:
            for top_ratio in [0.999, 0.99, 0.98, 0.97, 0.96, 0.9, 0.6]:
                test_time = 100
                _raw_result_of__mean = torch.empty(size=[test_time])
                _raw_result_of__std = torch.empty(size=[test_time])
                for test_count in range(test_time):
                    some_randn = torch.randn(size=[100, 10000], device='cuda')*scale_factor
                    _temp_result = _raw_log10_avg_safe__with_batch(some_randn, top_ratio=top_ratio)
                    
                    _the_mean = _temp_result.mean().cpu().item()
                    #assert _float_equal(_the_mean, -0.16, 0.02)
                    _raw_result_of__mean[test_count] = _the_mean
                    
                    _the_std = _temp_result.std().cpu().item()
                    #assert _the_std<0.02
                    _raw_result_of__std[test_count] = _the_std
                    pass
                    
                print(f"scale_factor={scale_factor:6}, top_ratio={top_ratio:.3f}, avg={_raw_result_of__mean.mean():.3f}, std={_raw_result_of__std.mean():.4f}")
                pass
            pass
        return 
    ____test____top_ratio_scan_____raw_log10_avg_safe__with_batch()
    pass
if "basic behavior test         come back later" and __DEBUG_ME__() and False:
    def ____test____basic_behavior_of_____raw_log10_avg_safe__with_batch():
        _input = torch.ones(size=[1,20])
        _input = _input + torch.randn_like(_input)*0.001
        assert _tensor_equal(_raw_log10_avg_safe__with_batch(_input), torch.tensor([0.]),epsilon=0.01)
        _input = torch.ones(size=[1,20])*-1.
        _input = _input + torch.randn_like(_input)*0.001
        assert _tensor_equal(_raw_log10_avg_safe__with_batch(_input), torch.tensor([0.]),epsilon=0.01)
        

        _input = torch.ones(size=[1,11])
        _input[0,0] = 0.
        assert _tensor_equal(_raw_log10_avg_safe__with_batch(_input), torch.tensor([0.]),epsilon=0.01)
        
        _input = torch.ones(size=[1,11])
        _input[0,0] = 1e-10
        _result = _raw_log10_avg_safe__with_batch(_input)
        assert _tensor_equal(_raw_log10_avg_safe__with_batch(_input), torch.tensor([0.]),epsilon=0.01)
        
        # 1w 继续。
        # 今天要做的。
        # 做成2个完全不同的版本，
        # 新版本不能输出batch。
        
        _input = torch.ones(size=[1,11])
        _input[0,0] = 1e-21
        _input[0,1] = 1e-10
        _result = _raw_log10_avg_safe__with_batch(_input)
        assert _tensor_equal(_raw_log10_avg_safe__with_batch(_input), torch.tensor([-1.]),epsilon=0.01)
        
        _input = torch.ones(size=[1,11])
        _input[0,0] = 1e-10
        _input[0,1] = 1e10
        assert _tensor_equal(_raw_log10_avg_safe__with_batch(_input), torch.tensor([1.]),epsilon=0.01)

        
        for _ in range(11):
            _rand_number = random.random()
            _input = torch.ones(size=[1,11])
            _input[0,0] = 1e-10
            _input[0,1] = 10**_rand_number
            assert _tensor_equal(_raw_log10_avg_safe__with_batch(_input), torch.tensor([_rand_number/10.]),epsilon=0.01)
            pass
        
        
        
        #batch>1
        _input = torch.ones(size=[6,11])
        _input[0] = _input[0] + torch.randn(size=[1,11])*0.001
        _input[1] = _input[1]*-1 + torch.randn(size=[1,11])*0.001
        _input[2,0] = 0.
        _input[3,0] = 1e-10
        _input[4,0] = 1e-21
        _input[4,1] = 1e-10
        _input[5,0] = 1e-10
        _input[5,1] = 1e10
        _answer = torch.tensor([0., 0., 0., 0., -1., 1.])
        a = _raw_log10_avg_safe__with_batch(_input)
        _result = _raw_log10_avg_safe__with_batch(_input)
        assert _tensor_equal(_raw_log10_avg_safe__with_batch(_input), _answer,epsilon=0.01)
        
        #about the stability
        _input = torch.randn(size=[1,10000])
        _ref_answer = _raw_log10_avg_safe__with_batch(_input)
        _input = torch.randn(size=[1,1000])
        assert _tensor_equal(_raw_log10_avg_safe__with_batch(_input), _ref_answer,epsilon=0.03)
        _input = torch.randn(size=[1,100000])
        a = _raw_log10_avg_safe__with_batch(_input)
        assert _tensor_equal(_raw_log10_avg_safe__with_batch(_input), _ref_answer,epsilon=0.018)
        
        _dim_list =     [1e2,   1e3,    1e4,    1e5,    1e6,  ]#  1e7] # the last one is too slow.
        _mean_list =    [-0.16, -0.16,  -0.16,  -0.16,  -0.16,]#  -0.2]
        _mean_epsilon = [0.005, 0.005,  0.005,  0.005,  0.002,]#  0.005]
        _std_max =      [0.05,  0.02,   0.015,   0.015,  0.04,]#   0.07]
        for ii in range(_dim_list.__len__()):
            _dim = int(_dim_list[ii])
            _input = torch.randn(size=[100,_dim], device='cuda')
            _output = _raw_log10_avg_safe__with_batch(_input)
            mean_of_output = _output.mean().cpu()
            _tensor_equal(mean_of_output.reshape([1]), [_mean_list[ii]], _mean_epsilon[ii])    
            std_of_output = _output.std().cpu()
            assert std_of_output.lt(_std_max[ii])
            #prin(f"{_dim}  {mean_of_output}  {std_of_output}")
            pass
        
        return 
    
    ____test____basic_behavior_of_____raw_log10_avg_safe__with_batch()
    pass

def log10_avg_safe__with_batch(input:torch.Tensor, careful_level:int|None=None)->torch.Tensor:
    '''based on the test in top ratio, this function adjust the result automatically. 
    It call the inner raw function twice.'''
    
    "result. diff from 0.99 to 0.999 is 0.010, "
    "                  0.9  to 0.99  is 0.099, "
    "                  0.6  to 0.9   is 0.184,   this holds across all the scale_factor from 1e-3 to 1e3."
    "this result is only for the randn. It's a bit different for rand."
    
    _the_0_6_result = _raw_log10_avg_safe__with_batch(input=input,top_ratio=0.6,careful_level=careful_level)
    _the_0_9_result = _raw_log10_avg_safe__with_batch(input=input,top_ratio=0.9,careful_level=careful_level)
    _diff = _the_0_9_result-_the_0_6_result
    _diff = _diff*(0.110/0.184)
    result = _the_0_9_result+_diff
    return result
if "test" and __DEBUG_ME__() and False:
    # scale_factor= 0.001, avg=-3.268, std=0.0145
    # scale_factor=   1.0, avg=-0.268, std=0.0147
    # scale_factor=1000.0, avg= 2.732, std=0.0143
    def ____test____log10_avg_safe__with_batch():
        for scale_factor in [1e-3,1.,1e3]:
            test_time = 100
            _raw_result_of__mean = torch.empty(size=[test_time])
            _raw_result_of__std = torch.empty(size=[test_time])
            for test_count in range(test_time):
                some_randn = torch.randn(size=[100, 10000], device='cuda')*scale_factor
                _temp_result = log10_avg_safe__with_batch(some_randn)
                
                _the_mean = _temp_result.mean().cpu().item()
                #assert _float_equal(_the_mean, -0.16, 0.02)
                _raw_result_of__mean[test_count] = _the_mean
                
                _the_std = _temp_result.std().cpu().item()
                #assert _the_std<0.02
                _raw_result_of__std[test_count] = _the_std
                pass
                
            print(f"scale_factor={scale_factor:6}, avg={_raw_result_of__mean.mean():.3f}, std={_raw_result_of__std.mean():.4f}")
            pass
        return
    ____test____log10_avg_safe__with_batch()
    pass

" ^^^ version 1.   ///   vvv version 2."
" ^^^ version 1.   ///   vvv version 2."
" ^^^ version 1.   ///   vvv version 2."

def ____avg_of_top____v2(input:torch.Tensor, top_ratio = 0.9, greater_true_smaller_false = True)->torch.Tensor:
    assert input.shape.__len__() == 1
    n_elements_needed = int(input.nelement()*top_ratio+0.5)
    if n_elements_needed < 1:
        n_elements_needed = 1
        pass
    
    _sorted = input.sort(descending=greater_true_smaller_false).values
    _before_mean = _sorted[:n_elements_needed]
    the_mean = _before_mean.mean()#last dim
    return the_mean

" ^^^ version 2.   ///   vvv version 3. the top k version"
" ^^^ version 2.   ///   vvv version 3. the top k version"
" ^^^ version 2.   ///   vvv version 3. the top k version"
" I only need to change this one function....."

def avg_of_top(input:torch.Tensor, top_ratio = 0.9, greater_true_smaller_false = True)->torch.Tensor:
    "v3 of this function."
    
    assert input.shape.__len__() == 1# or <=2????
    n_elements_needed = int(input.nelement()*top_ratio+0.5)
    if n_elements_needed < 1:
        n_elements_needed = 1
        pass
    
    _before_mean = input.topk(n_elements_needed, dim=-1, largest=greater_true_smaller_false,sorted=False).values
    the_mean = _before_mean.mean()#last dim
    return the_mean

if "test v2 vs v3" and False:
    def ____test____v2_vs_v3():
        if "are they the same.     don't run this." and False:
            assert False, "the log10_avg__how_similar is recursive????"
            # assertion only test. no print.
            for dim in [2,3,4,10,100,1000]:
                if dim <100:
                    test_time = 1000
                    pass
                else:
                    test_time = 100
                    pass
                for test_count in range(test_time):
                    input = torch.rand(size=[100])
                    result_v2 = ____avg_of_top____v2(input)
                    result_v3 = avg_of_top(input)
                    assert _tensor_equal(result_v2, result_v3)
                    if result_v2.ne(result_v3):
                        _result_is_valid, the_difference = log10_avg__how_similar(result_v2, result_v3)
                        assert the_difference>5.
                        pass
                    pass
                pass#for test_count
            
            pass#/ test
        
        if "performance test" and True:
            from pytorch_yagaodirac_v2.timeit_yagaodirac import timeit
            # device cpu
            # v2_time = [ 0.00108272669,  0.00149213650,  0.07766181688,  13.09609061965]
            # v3_time = [ 0.00066916173,  0.00080774480,  0.01565510492,  1.72210222969]
            # dim_list = [ 2.00000000,  100.00000000,  10000.00000000,  1000000.00000000]
            
            # device cuda
            # v2_time = [ 0.24382240805,  0.00649701804,  0.17482644962,  0.08778678276]
            # v3_time = [ 0.00196404489,  0.00213772431,  0.00546183334,  0.03368567326]
            # dim_list = [ 2.00000000,  100.00000000,  10000.00000000,  1000000.00000000]
            
            v2_time = []#don't modify this.
            v3_time = []#don't modify this.
            
            device = 'cuda'
            time_at_most = 0.2
            loop_time = 100
            dim_list =       [  2,   100,10000,int(1e6)]
            for outter_iter_count in range(dim_list.__len__()):
                dim = dim_list[outter_iter_count]
                print(device)
            
                input = torch.rand(size=[dim], device=device)
                def func_null():
                    for _ in range(loop_time):
                        pass
                    return 
                null_time,_ = timeit(func_null, time_at_most = time_at_most)
                del func_null
                
                def func_v2():
                    for _ in range(loop_time):
                        result_v2 = ____avg_of_top____v2(input)
                        pass
                    return 
                raw__result_v2,_ = timeit(func_v2, time_at_most = time_at_most)
                v2_time.append(raw__result_v2 - null_time)
                del func_v2
                
                def func_v3():
                    for _ in range(loop_time):
                        result_v3 = avg_of_top(input)
                        pass
                    return 
                raw__result_v3,_ = timeit(func_v3, time_at_most = time_at_most)
                v3_time.append(raw__result_v3 - null_time)
                del func_v3
                
                pass #for outter_iter_count
            print(f"device {device}")
            print(f"v2_time = {str_the_list(v2_time, 11)}")    
            print(f"v3_time = {str_the_list(v3_time, 11)}")    
            print(f"dim_list = {str_the_list(dim_list, 8)}")    
            
            pass#/ test
        
        return 
        
    ____test____v2_vs_v3()
    pass    

def avg_of_bottom(input:torch.Tensor, bottom_ratio = 0.9)->torch.Tensor:
    return avg_of_top(input, bottom_ratio, False)
if "test" and __DEBUG_ME__() and False:
    def ____test____avg_of_one_side():
        if "avg_of_top":
            for _ in range(123):
                _temp_random = torch.rand(size=[90])+1.
                _ref = _temp_random.mean()
                _temp_list = (_temp_random).tolist()
                _temp_list.extend((torch.rand(size=[10])).tolist())
                random.shuffle(_temp_list)
                random.shuffle(_temp_list)
                random.shuffle(_temp_list)
                input = torch.tensor(_temp_list)
                result = avg_of_top(input)
                assert _tensor_equal(result, _ref)
                pass
            pass
        
        if "avg_of_bottom":
            for _ in range(123):
                _temp_random = torch.rand(size=[90])
                _ref = _temp_random.mean()
                _temp_list = (_temp_random).tolist()
                _temp_list.extend((torch.rand(size=[10])+1.).tolist())
                random.shuffle(_temp_list)
                random.shuffle(_temp_list)
                random.shuffle(_temp_list)
                input = torch.tensor(_temp_list)
                result = avg_of_bottom(input)
                assert _tensor_equal(result, _ref)
                result = avg_of_top(input, greater_true_smaller_false = False)
                assert _tensor_equal(result, _ref)
                pass
            pass
        
        return 
    
    ____test____avg_of_one_side()
    pass

def _raw_log10_avg_safe(input:torch.Tensor, top_ratio = 0.9, recommended_gpu_device:torch.device = 'cuda')->torch.Tensor:
    '''check out the log10 measurement.py to see how this measurement function helps.
    
    This is the new version. 
    
    I found the new way to accelerate this function is always gpu+torch.sort
    So, if a data is on cpu, so move it to gpu and sort it and get the amount, avg, return.
    Yeah, the old my version retired. No need for a mask anymore.
    
    this function treat the entire tensor as a vector. It only returns 1 number as result.
    
    
    old docs below. 
    
    Calcs the average of log10 of abs of input. 
    
    The least log intermediate results are ignored. Because if a number is very close to 0, 
    the log10 of it is very negative, and any noise on such elements will introduce a bit 
    noise into the final result. So they are ignored.

    Inside this function, it calls get_mask_of_top_element__rough to help filter the
    bad intermediate results. That function also helps in a lot other cases.
    
    This function is mainly designed to help extract info from tensors, and to help
    measure some aspects in neural network training dynamics.
    
    If you don't like the shape convention and you know what you are doing, fell free
    to modify this function and the inner get_mask_of_top_element__rough function.
    '''
    #assert input.shape.__len__() <= 2, "my convention, shape is [batch, anything]"
    #ori_shape = input.shape
    
    if isinstance(recommended_gpu_device, str):
        assert recommended_gpu_device != 'cpu', "I tested, it's slower, or run any benchmark first and decide."
        pass
    else:
        assert recommended_gpu_device.type != 'cpu', "I tested, it's slower, or run any benchmark first and decide."
        pass
    
    ori_cpu = (input.device.type == 'cpu')
    input = input.to(device=recommended_gpu_device)
    with torch.no_grad():
        log_of_input = input.abs().log10()
        
        #safety
        #log only returns -torch.inf, torch.nan.    only +inf when input +inf. They all too wrong, let's remove them all.
        no_nan_log = log_of_input.nan_to_num( -999.,posinf=-999.,neginf=-999.) 
        _flag_wrong = no_nan_log.lt(-998.)
        if _flag_wrong.any():
            no_nan_log = no_nan_log[_flag_wrong.logical_not()]
            pass
        #safe_log is safe.
        #assert False
        
        the_mean = avg_of_top(no_nan_log.reshape([-1]), top_ratio=top_ratio)
        #old, now in a new function
        # n_elements_needed = int(input.nelement()*top_ratio+0.5)
        # if n_elements_needed < 1:
        #     n_elements_needed = 1
        #     pass
        # 
        # no_nan_log = no_nan_log.sort(descending=True).values[:n_elements_needed]
        # the_mean = no_nan_log.mean()#last dim
        
        #now data is on gpu.
        if ori_cpu:
            return the_mean.to('cpu')
        return the_mean
    pass#end of function
"Bc random.py imports this file. So this test is done here, with a function in random.py copy pasted here."
if "device adaption" and __DEBUG_ME__() and False:
    def ____test____log10_avg_safe____device_adaption():
        a = torch.tensor([1.])
        b = _raw_log10_avg_safe(a)
        assert a.device.type == 'cpu'
        assert a.shape == torch.Size([1])
        assert b.device.type == 'cpu'
        
        a = torch.tensor([[1.]], device='cuda')
        b = _raw_log10_avg_safe(a)
        assert a.device.type == 'cuda'
        assert a.shape == torch.Size([1,1])
        assert b.device.type == 'cuda'
        
        return 
    ____test____log10_avg_safe____device_adaption()
    pass
if "basic behavior test" and __DEBUG_ME__() and False:
    def ____test____basic_behavior_of____log10_avg_safe():
        if False:
            _input = torch.ones(size=[1,20])
            _input = _input + torch.randn_like(_input)*0.001
            assert _tensor_equal(_raw_log10_avg_safe(_input), torch.tensor([0.]),epsilon=0.01)
            _input = torch.ones(size=[1,20])*-1.
            _input = _input + torch.randn_like(_input)*0.001
            assert _tensor_equal(_raw_log10_avg_safe(_input), torch.tensor([0.]),epsilon=0.01)
            

            _input = torch.ones(size=[1,11])
            _input[0,0] = 0.
            assert _tensor_equal(_raw_log10_avg_safe(_input), torch.tensor([0.]),epsilon=0.01)
        
        _input = torch.ones(size=[1,11])
        _input[0,0] = 1e-10
        _result = _raw_log10_avg_safe(_input)
        assert _tensor_equal(_raw_log10_avg_safe(_input), torch.tensor([0.]),epsilon=0.01)
        
        # 1w 继续。
        # 今天要做的。
        # 做成2个完全不同的版本，
        # 新版本不能输出batch。
        
        _input = torch.ones(size=[1,11])
        _input[0,0] = 1e-21
        _input[0,1] = 1e-10
        _result = _raw_log10_avg_safe(_input)
        assert _tensor_equal(_raw_log10_avg_safe(_input), torch.tensor([-1.]),epsilon=0.01)
        
        _input = torch.ones(size=[1,11])
        _input[0,0] = 1e-10
        _input[0,1] = 1e10
        assert _tensor_equal(_raw_log10_avg_safe(_input), torch.tensor([1.]),epsilon=0.01)

        
        for _ in range(11):
            _rand_number = random.random()
            _input = torch.ones(size=[1,11])
            _input[0,0] = 1e-10
            _input[0,1] = 10**_rand_number
            assert _tensor_equal(_raw_log10_avg_safe(_input), torch.tensor([_rand_number/10.]),epsilon=0.01)
            pass
        
        
        # only the with batch version works with this.
        # #batch>1
        # _input = torch.ones(size=[6,11])
        # _input[0] = _input[0] + torch.randn(size=[1,11])*0.001
        # _input[1] = _input[1]*-1 + torch.randn(size=[1,11])*0.001
        # _input[2,0] = 0.
        # _input[3,0] = 1e-10
        # _input[4,0] = 1e-21
        # _input[4,1] = 1e-10
        # _input[5,0] = 1e-10
        # _input[5,1] = 1e10
        # _answer = torch.tensor([0., 0., 0., 0., -1., 1.])
        # a = log10_avg_safe(_input)
        # _result = log10_avg_safe(_input)
        # assert _tensor_equal(log10_avg_safe(_input), _answer,epsilon=0.01)
        #
        #about the stability
        # _input = torch.randn(size=[1,10000])
        # _ref_answer = log10_avg_safe(_input)
        # _input = torch.randn(size=[1,1000])
        # assert _tensor_equal(log10_avg_safe(_input), _ref_answer,epsilon=0.03)
        # _input = torch.randn(size=[1,100000])
        # a = log10_avg_safe(_input)
        # assert _tensor_equal(log10_avg_safe(_input), _ref_answer,epsilon=0.018)
        
        # _dim_list =     [1e2,   1e3,    1e4,    1e5,    1e6,  ]#  1e7] # the last one is too slow.
        # _mean_list =    [-0.16, -0.16,  -0.16,  -0.16,  -0.16,]#  -0.2]
        # _mean_epsilon = [0.005, 0.005,  0.005,  0.005,  0.002,]#  0.005]
        # _std_max =      [0.05,  0.02,   0.015,   0.015,  0.04,]#   0.07]
        # for ii in range(_dim_list.__len__()):
        #     _dim = int(_dim_list[ii])
        #     _input = torch.randn(size=[100,_dim], device='cuda')
        #     _output = log10_avg_safe(_input)
        #     mean_of_output = _output.mean().cpu()
        #     _tensor_equal(mean_of_output.reshape([1]), [_mean_list[ii]], _mean_epsilon[ii])    
        #     std_of_output = _output.std().cpu()
        #     assert std_of_output.lt(_std_max[ii])
        #     #prin(f"{_dim}  {mean_of_output}  {std_of_output}")
        #     pass
        
        return 
    
    ____test____basic_behavior_of____log10_avg_safe()
    pass
if "top ratio scan" and __DEBUG_ME__() and False:
    "result. diff from 0.99 to 0.999 is 0.018, "
    "                  0.9  to 0.99  is 0.097, "
    "                  0.6  to 0.9   is 0.183,   this holds across all the scale_factor from 1e-3 to 1e3."
    "this result is only for the randn. It's a bit different for rand."
    # prin(-3.273+3.255,-3.255+3.158,-3.158+2.975,)
    # prin(-0.273+0.255,-0.255+0.158,-0.158- 0.025,)
    # prin(2.727-2.745,2.745-2.842,2.842-3.025,)
    
    #randn
    # scale_factor= 0.001, top_ratio=0.999, avg=-3.273, std=0.000487
    # scale_factor= 0.001, top_ratio=0.990, avg=-3.255, std=0.000469
    # scale_factor= 0.001, top_ratio=0.980, avg=-3.240, std=0.000452
    # scale_factor= 0.001, top_ratio=0.970, avg=-3.227, std=0.000436
    # scale_factor= 0.001, top_ratio=0.960, avg=-3.215, std=0.000449
    # scale_factor= 0.001, top_ratio=0.900, avg=-3.158, std=0.000408
    # scale_factor= 0.001, top_ratio=0.600, avg=-2.975, std=0.000353
    # scale_factor=   1.0, top_ratio=0.999, avg=-0.273, std=0.000501
    # scale_factor=   1.0, top_ratio=0.990, avg=-0.255, std=0.000459
    # scale_factor=   1.0, top_ratio=0.980, avg=-0.240, std=0.000445
    # scale_factor=   1.0, top_ratio=0.970, avg=-0.227, std=0.000451
    # scale_factor=   1.0, top_ratio=0.960, avg=-0.215, std=0.000448
    # scale_factor=   1.0, top_ratio=0.900, avg=-0.158, std=0.000412
    # scale_factor=   1.0, top_ratio=0.600, avg= 0.025, std=0.000333
    # scale_factor=1000.0, top_ratio=0.999, avg= 2.727, std=0.000495
    # scale_factor=1000.0, top_ratio=0.990, avg= 2.745, std=0.000478
    # scale_factor=1000.0, top_ratio=0.980, avg= 2.760, std=0.000467
    # scale_factor=1000.0, top_ratio=0.970, avg= 2.773, std=0.000445
    # scale_factor=1000.0, top_ratio=0.960, avg= 2.785, std=0.000458
    # scale_factor=1000.0, top_ratio=0.900, avg= 2.842, std=0.000413
    # scale_factor=1000.0, top_ratio=0.600, avg= 3.025, std=0.000347
    
    def ____test____top_ratio_scan_____raw_log10_avg_safe():
        for scale_factor in [1e-3,1.,1e3]:
            #for top_ratio in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2 ,0.1]:
            #for top_ratio in [0.999, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.9, 0.8, 0.7, 0.6, 0.5]:
            for top_ratio in [0.999, 0.99, 0.98, 0.97, 0.96, 0.9, 0.6]:
                test_time = 1000
                _raw_result = torch.empty(size=[test_time])
                for test_count in range(test_time):
                    some_randn = torch.randn(size=[100, 10000], device='cuda')*scale_factor
                    _temp_result = _raw_log10_avg_safe(some_randn, top_ratio=top_ratio)
                    _raw_result[test_count] = _temp_result
                    pass
                    
                print(f"scale_factor={scale_factor:6}, top_ratio={top_ratio:.3f}, avg={
                    _raw_result.mean():.3f}, std={_raw_result.std():.6f}")
                pass
            pass
        return 
    ____test____top_ratio_scan_____raw_log10_avg_safe()
    pass

def log10_avg_safe(input:torch.Tensor)->torch.Tensor:
    '''based on the test in top ratio, this function adjust the result automatically. 
    It call the inner raw function twice.'''
    
    "result. diff from 0.99 to 0.999 is 0.018, "
    "                  0.9  to 0.99  is 0.097, "
    "                  0.6  to 0.9   is 0.183,   this holds across all the scale_factor from 1e-3 to 1e3."
    "this result is only for the randn. It's a bit different for rand."
    
    _the_0_6_result = _raw_log10_avg_safe(input=input,top_ratio=0.6)
    _the_0_9_result = _raw_log10_avg_safe(input=input,top_ratio=0.9)
    _diff = _the_0_9_result-_the_0_6_result
    _diff = _diff*(0.117/0.183)
    result = _the_0_9_result+_diff
    return result
if "test" and __DEBUG_ME__() and False:
    # scale_factor= 0.001, avg=-3.268, std=0.0145
    # scale_factor=   1.0, avg=-0.268, std=0.0147
    # scale_factor=1000.0, avg= 2.732, std=0.0143
    def ____test____log10_avg_safe():
        for scale_factor in [1e-3,1.,1e3]:
            test_time = 1000
            _raw_result = torch.empty(size=[test_time])
            for test_count in range(test_time):
                some_randn = torch.randn(size=[100, 10000], device='cuda')*scale_factor
                _temp_result = log10_avg_safe(some_randn)
                _raw_result[test_count] = _temp_result
                pass
                
            print(f"scale_factor={scale_factor:6}, avg={_raw_result.mean():.3f}, std={
                _raw_result.std():.6f}")
            pass
        return
    ____test____log10_avg_safe()
    pass

def log10_avg__how_similar(host:torch.Tensor, guest:torch.Tensor, )->tuple[bool, torch.Tensor]:
    '''return _result_is_valid, the_difference
    
    If the result is less than 1, don't trust it.
    
    result feels like log10(host) - log10(host-guest)
    
    The result has some error.'''
    useful_flag_1 = host.ne(0.)
    left_hand_side__useful = host[useful_flag_1]
    left_hand_side = log10_avg_safe(left_hand_side__useful)

    diff = host - guest
    useful_flag_2 = diff.ne(0.)
    right_hand_side__useful = diff[useful_flag_2]
    right_hand_side = log10_avg_safe(right_hand_side__useful)
    
    if left_hand_side.isnan() or right_hand_side.isnan():
        return (False,torch.empty(size=[]))
    return (True, left_hand_side - right_hand_side)
    #end of function
    
# host = torch.randn(size=[3,3])#tensor([1.,2,3])
# guest = torch.zeros_like(host)#host.detach().clone()
# aaa = log10_avg__how_similar(host, guest)
# fds=432
    
if "test" and __DEBUG_ME__() and False:
    def ____test____log10_avg_diff_safe():
        #this one is weird.
        if "between 2 randn" and False:
            #result
            # diff__min   = [-2.691, -2.726, -1.856, -1.410, -0.892, -0.333, -0.215, -0.168]
            # diff__max   = [ 3.355,  2.295,  2.018,  1.152,  0.560,  0.038, -0.086, -0.136]
            # diff__avg   = [-0.114, -0.133, -0.143, -0.149, -0.152, -0.152, -0.151, -0.151]
            # dim_list   = [ 1.000,  2.000,  3.000,  5.000,  10.000,  100.,  1000.,  10000]
            
            diff__min = []#don't modify this.
            diff__max = []#don't modify this.
            diff__avg = []#don't modify this.
            
            #----------------#----------------#----------------
            dim_list =         [1,   2,   3,   5,  10,  100, 1000, 10000]
            test_time_list = [1000,1000,1000,1000,1000,1000, 1000,  1000,]
            for outter_param_count in range(dim_list.__len__()):
                dim = dim_list[outter_param_count]
                test_time = test_time_list[outter_param_count]
                print(test_time)
            #----------------#----------------#----------------
                _raw_result = torch.empty(size=[test_time])
                for test_count in range(test_time):
                    #----------------#----------------#----------------
                    host = torch.randn(size=[dim])
                    guest = torch.randn(size=[dim])
                    # this tool actually measures this --->> ^^^
                    _result_is_valid, the_difference = log10_avg__how_similar(host=host, guest=guest)
                    if not _result_is_valid:
                        continue
                    #----------------#----------------#----------------
                    _raw_result[test_count] = the_difference
                    pass
                diff__min.append(_raw_result.min())
                diff__max.append(_raw_result.max())
                diff__avg.append(_raw_result.mean())
                pass#for dim
            print(f"diff__min   = {str_the_list(diff__min    , 3)}")
            print(f"diff__max   = {str_the_list(diff__max    , 3)}")
            print(f"diff__avg   = {str_the_list(diff__avg    , 3)}")
            print(f"dim_list   = {str_the_list(dim_list     , 3)}")
            pass#/test
        
        if "scan the scaling factor" and False:
            #result
            # if dim == anything:
            # diff__max    = [ 3.000,  2.000,  1.000,  0.000, -1.000, -2.000, -3.000]
            # diff__avg    = [ 3.000,  2.000,  1.000,  0.000, -1.000, -2.000, -3.000]
            # scaling_factor= [-3.000, -2.000, -1.000,  0.000,  1.000,  2.000,  3.000]
            # I know it looks fake, but yeah.
            #----------------#----------------#----------------
            dim_list =         [1,   2,   3,   5,  10,  100, 1000, 10000]
            test_time_list = [1000,1000,1000,1000,1000,1000, 1000, 1000]
            for outter_param_count in range(dim_list.__len__()):
                dim = dim_list[outter_param_count]
                test_time = test_time_list[outter_param_count]
            #----------------#----------------#----------------
                
                diff__max = []#don't modify this.
                diff__avg = []#don't modify this.
                
                #----------------#----------------#----------------
                scaling_factor_list_as_pow = torch.linspace(-3.,3.,7)
                scaling_factor_list = torch.tensor(10.).pow(scaling_factor_list_as_pow)
                for scaling_factor in scaling_factor_list:
                    #scaling_factor = scaling_factor_list[inner_param_count]
                #----------------#----------------#----------------
                    
                    _raw_result = torch.empty(size=[test_time])
                    for test_count in range(test_time):
                        
                        #----------------#----------------#----------------
                        host = torch.randn(size=[dim])
                        guest = host.detach().clone()*(1.+scaling_factor)
                        _result_is_valid, the_difference = log10_avg__how_similar(host=host, guest=guest)
                        if not _result_is_valid:
                            continue
                        #----------------#----------------#----------------
                        _raw_result[test_count] = the_difference
                        pass
                    diff__max.append(_raw_result.max())
                    diff__avg.append(_raw_result.mean())
                    pass#for dim(the inner)
                print(f"if dim == {dim}:")
                print(f"diff__max    = {str_the_list(diff__max    , 3)}")
                print(f"diff__avg    = {str_the_list(diff__avg    , 3)}")
                print(f"scaling_factor= {str_the_list(scaling_factor_list_as_pow     , 3)}")
                print("pass")
                pass#for outter_param_count
                
            pass#/test
    
    ____test____log10_avg_diff_safe()
    
    pass



