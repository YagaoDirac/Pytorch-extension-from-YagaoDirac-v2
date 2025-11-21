'''how to import:
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

import test_tool.simple_data_analysis as [name it]
'''

import statistics



def print_list_stats(inp:list[float])->str:
    if inp.__len__() == 0:
        return "No data"
    if inp.__len__() == 1:
        return f'{inp[0]} (1 sample)'
    return f'(mean,median,std) {statistics.mean(inp)} , {statistics.median(inp)} , {statistics.stdev(inp)} , ({inp.__len__()} samples)'
if "test" and False:
    empty_str = print_list_stats([])
    a_lil_str = print_list_stats([1])
    a_lot_str = print_list_stats([1,2])
    a_lot_str_2 = print_list_stats([1,2,3,2,2])
    pass


def print_list_stats__int(inp:list[int])->str:
    if inp.__len__() == 0:
        return "No data"
    if inp.__len__() == 1:
        return f'{inp[0]} (1 sample)'
    return f'(mean,median,std) {statistics.mean(inp)} , {statistics.median(inp)} , {statistics.stdev(inp)} , ({inp.__len__()} samples)'


class first_time_epoch_tracker():
    original_loss:float
    from_1_over_10_to_the_power_of_n:int
    to_1_over_10_to_the_power_of_n:int
    
    _max_possible_pos:int
    _current_pos:int
    data:list[list[int]]#[power][index]
    def __init__(self, original_loss:float, from_1_over_10_to_the_power_of_n:int, 
                                        to_1_over_10_to_the_power_of_n:int):
        assert original_loss>0.
        assert 0<from_1_over_10_to_the_power_of_n
        assert from_1_over_10_to_the_power_of_n<to_1_over_10_to_the_power_of_n
        
        self.original_loss = original_loss
        self.from_1_over_10_to_the_power_of_n = from_1_over_10_to_the_power_of_n
        self.to_1_over_10_to_the_power_of_n = to_1_over_10_to_the_power_of_n
        
        self._max_possible_pos = to_1_over_10_to_the_power_of_n-from_1_over_10_to_the_power_of_n
        self._current_pos = 0
        self.data = []
        for _ in range(self._max_possible_pos+1):#+1
            self.data.append([])
        pass
    def next_round(self):
        self._current_pos = 0
        pass
    def clear(self):
        self._current_pos = 0
        self.data = []
        for _ in range(self._max_possible_pos+1):#+1
            self.data.append([])
        pass
    def _get_loss_standard(self, index:int)->float:
        result = self.original_loss * 0.1**(self.from_1_over_10_to_the_power_of_n+index)
        return result
    def try_add(self, epoch:int, loss:float):
        while True:
            the_standard = self._get_loss_standard(self._current_pos)
            if loss<the_standard:
                self.data[self._current_pos].append(epoch)
                self._current_pos +=1
                continue
            else:
                break
            #no tail
            pass
        pass
    def print(self)->str:
        result = ""
        for ii in range(self.data.__len__()):
            result += f'''reached loss {self._get_loss_standard(ii)}, {print_list_stats__int(self.data[ii]) }'''
            if self.data[ii].__len__()>0:
                result += f''' / {self.data[ii].__len__()} sample(s) in total\n'''
                pass
            else:
                result += f'''\n'''
                break
            pass
        return result
    #end of class
if "test" and False:
    ftet = first_time_epoch_tracker(1.0, 2, 5)
    assert ftet._get_loss_standard(0) == 1.0*(0.1**2)    
    assert ftet._get_loss_standard(1) == 1.0*(0.1**3)    
    
    
    ftet = first_time_epoch_tracker(1.0, 2, 5)
    assert ftet.data.__len__() == 4
    _the_str = ftet.print()
    assert _the_str == f'''reached loss {1.0*(0.1**2)}, {print_list_stats__int([])}\n'''
                
    
    ftet = first_time_epoch_tracker(1.0, 2, 5)
    ftet.try_add(3, 0.015)
    assert ftet.data[0].__len__() == 0
    assert ftet._current_pos == 0
    _the_str = ftet.print()
    assert _the_str == f'''reached loss {1.0*(0.1**2)}, {print_list_stats__int([])}\n'''
    
    ftet.try_add(5, 0.005)
    assert ftet.data[0].__len__() == 1
    assert ftet.data[1].__len__() == 0
    assert ftet._current_pos == 1
    _the_str = ftet.print()
    assert _the_str == f'''reached loss {1.0*(0.1**2)}, {print_list_stats__int([5]) \
                } / {1} sample(s) in total\n''' + \
                    f'''reached loss {1.0*(0.1**3)}, {print_list_stats__int([])}\n'''
    
    ftet.try_add(15, 0.0005)
    assert ftet.data == [[5],[15],[],[]]
    ftet.try_add(25, 0.0005)
    assert ftet.data == [[5],[15],[],[]]
    ftet.next_round()
    ftet.try_add(33, 0.005)
    assert ftet.data == [[5,33],[15],[],[]]
    _the_str = ftet.print()
    assert _the_str == f'''reached loss {1.0*(0.1**2)}, {print_list_stats__int([5,33]) \
                } / {2} sample(s) in total\n''' + \
                    f'''reached loss {1.0*(0.1**3)}, {print_list_stats__int([15]) \
                } / {1} sample(s) in total\n''' + \
                    f'''reached loss {1.0*(0.1**4)}, {print_list_stats__int([])}\n'''
    
    ftet.clear()
    assert ftet.data == [[],[],[],[]]
    pass



