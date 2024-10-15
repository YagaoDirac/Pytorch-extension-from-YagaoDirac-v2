from typing import List
import math

class Print_Timing:
    r'''
    >>> pt = Print_Timing()
    >>> for i in range(501):
    >>>     if pt.check(i):
    >>>         print(i, end = ", ")
    >>>         pass
    >>>     pass
    The result is 0, 1, 2, 5, 10, 19, 34, 62, 100, 200, 300, 400, 500, 
    '''
    def __init__(self, start_with = 0, first = 3, max_gap = 100, density:float = 4.):
        self.start_with = start_with
        self.first = first
        self.max_gap = max_gap
        
        self.return_true_when:List[float] = []
        the_exp = math.log10(first-start_with-1)
        end_log = math.log10(max_gap)
        invert_of_density = 1/float(density)
        while the_exp<end_log:
            self.return_true_when.append(int(math.pow(10, the_exp)))
            the_exp += invert_of_density
            pass
        pass
    #end of function
    
    def check(self, epoch:int)->bool:
        if epoch>=self.max_gap and epoch%self.max_gap==0:
            return True
        
        calibrated_epoch = epoch-self.start_with+1
        if calibrated_epoch<=self.first:
            return True
        if calibrated_epoch in self.return_true_when:
            return True
        return False
    #end of function
            
    pass# end of class

pt = Print_Timing()
for i in range(501):
    if pt.check(i):
        print(i, end = ", ")
        pass
    pass














