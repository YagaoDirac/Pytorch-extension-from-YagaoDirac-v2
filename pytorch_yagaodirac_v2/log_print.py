'''
to import this tool:
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent maybe multiple times.))
'''
from pathlib import Path

class Log:
    def __init__(self, log_name:str):
        self.log_name = log_name+".log"
        pass
    def log_print(self, content:str, end='\n'):
        print(content, end = end)
        with open(self.log_name,mode="a") as file:
            file.write(content)
            file.write(end)
            pass
        pass

if "basic test" and False:
    log = Log("log name")
    log.log_print("test content.")
    log.log_print("test content.")
    log.log_print("test content. no new line.", end="")
    log.log_print("test content. no new line.", end="")
    log.log_print("test content. no new line.", end="")
