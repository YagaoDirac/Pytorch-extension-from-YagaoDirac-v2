import time
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
import torch
def __DEBUG_ME__()->bool:
    return __name__ == "__main__"
def _line_():
    caller_s_frame = sys._getframe(1)
    caller_s_line_number = caller_s_frame.f_lineno
    assert caller_s_line_number is not None
    return caller_s_line_number#######
from pytorch_yagaodirac_v2.Util import str_the_list






if "test" and __DEBUG_ME__() and True:
    def ____test____by_dim():
        
        if "by dim":
            if True:
                #result
                pass
            
            print(f"__LINE__ {_line_()}    by dim")
            
            xxxxxxx = []#don't modify this
            #------------------#------------------#------------------
            dim_list =                          [  2,   10,   100, 1000]
            number_of_tests_list = torch.tensor([1000,10000, 5000, 2000])
            number_of_tests_list = number_of_tests_list.mul(1.).to(torch.int32)
            for outter_param_set in range(dim_list.__len__()):
                dim = dim_list[outter_param_set]
                # iota_of_dim = iota(dim)
                number_of_tests = number_of_tests_list[outter_param_set]
                device = 'cpu'
                # if dim>100:
                #     device = 'cuda'
                #     pass
                print(f"dim {dim}   test_time {number_of_tests}    device {device}")
            #------------------#------------------#------------------

                _raw_result__xxxxxxx = torch.empty(size=[number_of_tests])
                _when_start = time.perf_counter()
                
                for ii__test in range(number_of_tests):
                    
                    #------------------#------------------#------------------
                    #<  init

                    #<  measure
                    _this_result = 123
                    #------------------#------------------#------------------
                    
                    _raw_result__xxxxxxx[ii__test] = _this_result
                    pass#for ii__test
                _when_end = time.perf_counter()
                print(f"{device}   {_when_end - _when_start:.6f} , or {(_when_end - _when_start)/number_of_tests:.6f} per test")
                
                xxxxxxx.append(_raw_result__xxxxxxx.mean())
                pass#for outter_param_set
            
            print(f"xxxxxxx   = {str_the_list(xxxxxxx  , 3)}")
            print(f"dim        = {str_the_list(dim_list, 0, ",    ")}")
            
            pass#/ test
        
        return
    
    ____test____by_dim()
    
    
    def ____test____1_param():
        
        if "scan 1 param" and False:
            print(f"__LINE__ {_line_()}    scan 1 param")
            
            #------------------#------------------#------------------
            dim_list =                          [ 2,  10,100, 1000]
            number_of_tests_list = torch.tensor([100,100, 50,  10])
            number_of_tests_list = number_of_tests_list.mul(1.).to(torch.int32)
            for ii_outter_param_set in range(dim_list.__len__()):
                dim = dim_list[ii_outter_param_set]
                # iota_of_dim = iota(dim)
                number_of_tests = int(number_of_tests_list[ii_outter_param_set].item())
                device = 'cpu'
                # if dim>100:
                #     device = 'cuda'
                #     pass
                print(f"dim {dim}   test_time {number_of_tests}    device {device}")
            #------------------#------------------#------------------
                xxxxxxxxxxxxxxxx = []#don't modify this.
                
                scanned_param_list = []################################################
                _when_start = time.perf_counter()
                
                for scanned_param in scanned_param_list:
                    _raw_result__xxxxxxx = torch.empty(size=[number_of_tests])
                    for ii__test in range(number_of_tests):
                        
                        #------------------#------------------#------------------
                        #<  init           
                        
                        #<  measure
                        _this_result = 123
                        #------------------#------------------#------------------
                        
                        _raw_result__xxxxxxx[ii__test] = _this_result
                        pass#for ii__test
                    
                    #scanned_param_list.append()
                    
                    pass#for scanned_param
                _when_end = time.perf_counter()
                
                print(f"{device}   {_when_end - _when_start:.6f} , or {(_when_end - _when_start)/number_of_tests:.6f} per test")
                print(f"dim {dim}")
                print(f"scanned_param_list = {str_the_list(scanned_param_list, 3)}")#########################
                print(f"xxxxxxxxxxxxxxxx = {str_the_list(xxxxxxxxxxxxxxxx, 3)}")#########################
                ################################
                pass#for ii_outter_param_set
            pass#/ test
        
        return 
    
    ____test____1_param()()
    
    pass


'''
xxxxx_list = []
for xxx_param_set in range(xxxxx_list.__len__()):
    xxxxx = xxxxx_list[xxx_param_set]
'''






    

    
    
    
    
    
    
    
    