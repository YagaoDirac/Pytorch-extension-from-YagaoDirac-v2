import torch


class test_class_for_digital_mapper_v2_3(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        out_features = 3
        in_features = 4
        
        self.raw_weight_o_i = torch.nn.Parameter(torch.randn([out_features, in_features]), requires_grad=False)
        self.raw_weight_o_i.data = torch.tensor([[-2.,3,1,-1,],[-4.,5,1,-1,],
                                                 [-3.,-2,0,0,],])
        pass
    
    def run(self):
        # rand_str = pos_offset = neg_offset = 0
        rand_str = 0.01
        pos_offset = 0.1
        neg_offset = -0.1
                
        self.raw_weight_o_i+=torch.randn_like(self.raw_weight_o_i.data)*rand_str
        
        self.raw_weight_o_i[0,0]+=torch.rand([1])[0]*pos_offset
        self.raw_weight_o_i[0,1]+=torch.rand([1])[0]*neg_offset
        
        #now protect it.
        
        #if max of each is less than 1, offset it to have a max of 1.
        max_element = self.raw_weight_o_i.max(dim = 1,keepdim=True).values
        dist_to_1_raw = 1-max_element
        dist_to_1 = torch.max(dist_to_1_raw,torch.zeros([1]))
        #print(self.raw_weight_o_i.data)
        self.raw_weight_o_i += dist_to_1
        
        #compress anything greater than 1.
        row_gt_1:torch.Tensor = max_element.gt(1.)
        element_gt_0:torch.Tensor = self.raw_weight_o_i.gt(0.)
        needs_shrink_for_pos = row_gt_1.expand([-1,element_gt_0.shape[1]]).logical_and(element_gt_0)
        div_this_for_pos = needs_shrink_for_pos*max_element+needs_shrink_for_pos.logical_not()
        #print(self.raw_weight_o_i.data)
        self.raw_weight_o_i.data = self.raw_weight_o_i/div_this_for_pos
        
        #compress anything less than -1
        
        min_element = self.raw_weight_o_i.min(dim = 1,keepdim=True).values
        row_lt_neg_1:torch.Tensor = min_element.lt(-1.)
        element_lt_0:torch.Tensor = self.raw_weight_o_i.lt(0.)
        needs_shrink_for_neg = row_lt_neg_1.expand([-1,element_gt_0.shape[1]]).logical_and(element_lt_0)
        div_this_for_neg = needs_shrink_for_neg*min_element*-1.+needs_shrink_for_neg.logical_not()
        #print(self.raw_weight_o_i.data)
        self.raw_weight_o_i.data = self.raw_weight_o_i/div_this_for_neg
        #print(self.raw_weight_o_i.data)
        
        fds=432
        pass
    def print(self):
        print(self.raw_weight_o_i.data[0])
    pass
        
obj = test_class_for_digital_mapper_v2_3()
obj.print()
for _ in range(100):
    for _ in range(10):
        obj.run()
        obj.print()