import os, sys
ori_path = sys.path[0]
index = ori_path.rfind("\\")
upper_folder = ori_path[:index]
sys.path.append(upper_folder)
del ori_path
del index
del upper_folder

from pytorch_yagaodirac_v2 import util



# I found this from internet. 
sys.path.append(os.getcwd())

