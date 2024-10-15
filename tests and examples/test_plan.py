from typing import Callable
class Test_Config:
    def __init__(self, log_file:str = "out.txt"):
        self.filename = log_file
        pass
    
    def add_config(self, config_name)
    
    def run(self, test_func:Callable):
        test_func