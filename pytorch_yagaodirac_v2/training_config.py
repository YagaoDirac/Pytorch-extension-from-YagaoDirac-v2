from typing import Optional

class Training_Config():
    pass

class Training_Config_fixed_steps(Training_Config):
    def __init__(self, epochs:int, including_0=True):
        super().__init__()
        self.epochs = epochs
        self.including_0 = including_0
        pass
    def check(self, current_epoch:int)->bool:
        if self.including_0:
            return current_epoch>=self.epochs
        else:
            return current_epoch>self.epochs
    pass

if 'basic test' and False:
    tc_fixed = Training_Config_fixed_steps(10)
    print(tc_fixed.check(5))
    print(tc_fixed.check(9))
    print(tc_fixed.check(10))
    tc_fixed = Training_Config_fixed_steps(10,False)
    print(tc_fixed.check(9))
    print(tc_fixed.check(10))
    print(tc_fixed.check(11))
    pass

class Training_Config_proportion_of_loss(Training_Config):
    def __init__(self, proportion:float):
        super().__init__()
        self.proportion = proportion
        self.target_loss:Optional[float] = None
        pass
    def set_init_loss(self, init_loss:float):
        self.target_loss = init_loss*self.proportion
        pass
    def check(self, current_loss:float)->bool:
        if self.target_loss is None:
            raise Exception("Call set_init_loss before use.")
        return current_loss<self.target_loss
    pass

if 'basic test' and False:
    tc_loss = Training_Config_proportion_of_loss(0.2)
    #print(tc_loss.check(0.05))
    tc_loss.set_init_loss(0.1)
    print(tc_loss.check(0.05))
    print(tc_loss.check(0.021))
    print(tc_loss.check(0.02))
    pass