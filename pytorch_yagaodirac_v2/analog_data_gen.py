from typing import Any, List, Optional, Self
import torch
import math


# input = torch.rand([data_amount, in_feature])+0.3
# #target = torch.rand([data_amount, 1])+0.1
# target = (input.pow(1.5).sum(dim=1)*0.05+0.).unsqueeze(1)




# input = torch.rand([data_amount, in_feature])+0.3
# # target = torch.rand([data_amount, 1])+0.1
# target = ((input.pow(1.5)-input.pow(2.5)).sum(dim=1)).unsqueeze(1)




# input = torch.rand([data_amount, in_feature])+0.3
# #target = input.pow(1.5)
# target = torch.rand([data_amount, 1])+0.1




# input = torch.rand([data_amount, in_feature])+0.3
# #target = input.pow(1.5)
# #target = torch.rand([data_amount, 1])+0.1
# target = ((input.pow(1.5)-input.pow(2.5)).sum(dim=1)).unsqueeze(1)





# input = torch.ones([data_amount, in_feature])*0.#as close to 0 as possible.
# target = torch.ones([data_amount, 1])






# input = (torch.rand([data_amount, in_feature])*0.9+0.1).pow(torch.rand([data_amount, 1])*15)
# #target = input.pow(1.5)
# target = (torch.rand([data_amount, 1])*0.9+0.1).pow(torch.rand([data_amount, 1])*15)




# input = (torch.rand([data_amount, in_feature])*0.9+0.1).pow(torch.rand([data_amount, 1])*15)
# #target = input.pow(1.5)
# target = (torch.rand([data_amount, 1])*0.9+0.1).pow(torch.rand([data_amount, 1])*15)




# input = (torch.rand([data_amount, in_feature])*0.9+0.1).pow(torch.rand([data_amount, 1])*15)
# #target = input.pow(1.5)
# target = (torch.rand([data_amount, 1])*0.9+0.1).pow(torch.rand([data_amount, 1])*15)



# input = torch.ones([data_amount, in_feature])*0.00001
# target = torch.ones([data_amount, 1])*0.00001



# input = (torch.rand([data_amount, in_feature])*0.9+0.1).pow(torch.rand([data_amount, 1])*15)
# #target = input.pow(1.5)
# target = (torch.rand([data_amount, 1])*0.9+0.1).pow(torch.rand([data_amount, 1])*15)




# input = (torch.rand([data_amount, in_feature])*0.9+0.1).pow(torch.rand([data_amount, 1])*15)
# #target = input.pow(1.5)
# target = (torch.rand([data_amount, 1])*0.9+0.1).pow(torch.rand([data_amount, 1])*15)+0.0001



#     input = (torch.rand([data_amount, in_feature])*0.9+0.1).pow(torch.rand([data_amount, 1])*15)
#     #target = input.pow(1.5)
#     target = (torch.rand([data_amount, 1])*0.9+0.1).pow(torch.rand([data_amount, 1])*15)




#     input = torch.rand([data_amount, in_feature])+0.3
#     #target = input.pow(1.5)
#     #target = torch.rand([data_amount, 1])+0.1
#     target = ((input.pow(1.5)-input.pow(2.5)).sum(dim=1)).unsqueeze(1)




