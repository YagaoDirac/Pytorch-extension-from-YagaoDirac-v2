


class DigitalMapperFunction_v2(torch.autograd.Function):
    r'''
    forward input list:
    >>> x = args[0]# shape must be [batch, in_features]
    >>> raw_weight:torch.Tensor = args[1]# shape must be [out_features, in_features], must requires grad.
    backward input list:
    >>> g_in #shape of g_in must be [batch, out_features]
    '''
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any)->Any:
        x = args[0]# shape must be [batch, in_features]
        raw_weight:torch.Tensor = args[1]# shape must be [out_features, in_features]
                
        #print(out_features_iota, "out_features_iota")
        index_of_max_o = raw_weight.max(dim=1).indices
        #print(index_of_max_o, "index_of_max_o")
        output:torch.Tensor
        output = x[:, index_of_max_o]
        
        '''Because raw_weight always requires grad, but the output is 
        calculated with x[:, index_of_max_o], which is unlike other multiplication or
        addition, if any of the input tensor/parameter requires grad, the result requires grad.
        In this case, the output should always require grad, but the program doesn't infer 
        this from the "x[:, index_of_max_o]", it only inherits from the x.
        So, I have to manually modify it.
        '''
        output.requires_grad_()

        out_features_s = torch.tensor(raw_weight.shape[0], device=raw_weight.device)
        raw_weight_shape = torch.tensor(raw_weight.shape, device=raw_weight.device)
        x_needs_grad = torch.tensor([x.requires_grad], device=x.device)
        ctx.save_for_backward(x, index_of_max_o, out_features_s, raw_weight_shape, x_needs_grad)
        return output

    @staticmethod
    def backward(ctx, g_in):
        #shape of g_in must be [batch, out_features]
        
        
        x:torch.Tensor
        index_of_max_o:torch.Tensor
        out_features_s:torch.Tensor
        raw_weight_shape:torch.Tensor
        x_needs_grad:torch.Tensor
        x, index_of_max_o, out_features_s, raw_weight_shape, x_needs_grad = ctx.saved_tensors
            
        # print(g_in, "g_in", __line__str())
            
        input_reshaped_b_1_i = x[:,None,:]
        # print(input_reshaped_b_1_i, "input_reshaped_b_1_i")
        #print(input_reshaped_b_1_i.shape, "input_reshaped_b_1_i.shape")
        g_in_reshaped_b_o_1:torch.Tensor = g_in[:,:,None]
        #print(g_in_reshaped_b_o_1, "g_in_reshaped_b_o_1")
        #print(g_in_reshaped_b_o_1.shape, "g_in_reshaped_b_o_1.shape")
        grad_for_raw_weight_before_sum_b_o_i = g_in_reshaped_b_o_1.matmul(input_reshaped_b_1_i)
        # print(grad_for_raw_weight_before_sum_b_o_i, "grad_for_raw_weight_before_sum_b_o_i before sum")
        #print(grad_for_raw_weight_before_sum_b_o_i.shape, "grad_for_raw_weight_before_sum_b_o_i.shape before sum")
        grad_for_raw_weight_o_i = grad_for_raw_weight_before_sum_b_o_i.sum(dim=0, keepdim=False)
        # print(grad_for_raw_weight_o_i, "grad_for_raw_weight_o_i after sum, the final result.")
        #print(grad_for_raw_weight_o_i.shape, "grad_for_raw_weight_o_i.shape. Should be ", raw_weight.shape)

        if x_needs_grad.logical_not():
            return None, grad_for_raw_weight_o_i

        in_features_s = x.shape[1]
        out_features_iota_o = torch.linspace(0, out_features_s-1, out_features_s, dtype=torch.int32)

        one_hot_o_i = torch.zeros(raw_weight_shape[0], raw_weight_shape[1], device=raw_weight_shape.device)
        one_hot_o_i[out_features_iota_o, index_of_max_o] = 1.
        # print(g_in, "g_in")
        # print(one_hot_o_i, "one_hot_o_i")
        one_hot_expanded_fake_b_o_i = one_hot_o_i.expand(g_in.shape[0], -1, -1)
        #print(one_hot_expanded_fake_b_o_i, "one_hot_expanded_fake_b_o_i")
        #print(one_hot_expanded_fake_b_o_i.shape, "one_hot_expanded_fake_b_o_i.shape")

        g_in_reshaped_expanded_b_o_fake_i = g_in_reshaped_b_o_1.expand(-1, -1, in_features_s)
        #print(g_in_reshaped_expanded_b_o_fake_i, "g_in_reshaped_expanded_b_o_fake_i")
        #print(g_in_reshaped_expanded_b_o_fake_i.shape, "g_in_reshaped_expanded_b_o_fake_i.shape")

        one_hot_mul_g_in_b_o_i = one_hot_expanded_fake_b_o_i.mul(g_in_reshaped_expanded_b_o_fake_i)
        # print(one_hot_mul_g_in_b_o_i, "one_hot_mul_g_in_b_o_i")

        grad_for_x_b_i = one_hot_mul_g_in_b_o_i.sum(dim=1, keepdim=False)
        # print(grad_for_x_b_i, "grad_for_x_b_i")
        #print(grad_for_x_b_i.shape, "grad_for_x_b_i.shape")
                
        return grad_for_x_b_i, grad_for_raw_weight_o_i

    pass  # class

# '''multi layer grad backward test.'''
# x = torch.tensor([[1.11]])
# w1 = torch.tensor([[1.21], [1.22], ], requires_grad=True)
# w2 = torch.tensor([[1.31, 1.32]], requires_grad=True)
# pred1 = DigitalMapperFunction_v2.apply(x, w1)
# print(pred1, "pred1")
# pred_final:torch.Tensor = DigitalMapperFunction_v2.apply(pred1, w2)
# print(pred_final, "pred_final")
# g_in = torch.ones_like(pred_final.detach())
# #torch.autograd.backward(pred_final, g_in, inputs=[w1, w2])
# pred_final.backward(g_in)
# print(w1.grad, "w1.grad")
# print(w2.grad, "w2.grad")
# fds=423


# '''This basic test is a bit long. It consists of 3 parts. 
# A raw computation, using x, raw_weight, and g_in as input, calcs the output and both grads.
# Then, the equivalent calc, which use the proxy weight instead of raw_weight,
# which allows directly using autograd.backward function to calc the grads.
# At last, the customized backward version.
# They all should yield the same results.'''

# x = torch.tensor([[5., 6., 7.], [8., 9., 13]])
# in_features_s = x.shape[1]

# raw_weight = torch.tensor([[1., 2., 3.], [4., 2., 3.], [4., 5., 8.], [6., 2., 9.],[6., 2., 9.], ])
# out_features_s = raw_weight.shape[0]
# out_features_iota_o = torch.linspace(0, out_features_s-1, out_features_s, dtype=torch.int32)
# #print(out_features_iota, "out_features_iota")
# index_of_max_o = raw_weight.max(dim=1).indices
# print(index_of_max_o, "index_of_max_o")
# output = x[:, index_of_max_o]
# print(output, "output")

# #fake_ctx = (x, index_of_max_o, out_features_s)

# g_in = torch.tensor([[0.30013, 0.30103, 0.30113, 0.31003, 0.31013], [0.40013, 0.40103, 0.40113, 0.41003, 0.41013], ])
# print(output.shape, "output.shape, should be ", g_in.shape)
# #grad_for_raw_weight = torch.zeros_like(raw_weight)

# #sum_of_input = x.sum(dim=0, keepdim=True)
# input_reshaped_b_1_i = x[:,None,:]#x.unsqueeze(dim=1)
# print(input_reshaped_b_1_i, "input_reshaped_b_1_i")
# print(input_reshaped_b_1_i.shape, "input_reshaped_b_1_i.shape")
# g_in_reshaped_b_o_1 = g_in[:,:,None]#.unsqueeze(dim=-1)
# print(g_in_reshaped_b_o_1, "g_in_reshaped_b_o_1")
# print(g_in_reshaped_b_o_1.shape, "g_in_reshaped_b_o_1.shape")
# grad_for_raw_weight_before_sum_b_o_i = g_in_reshaped_b_o_1.matmul(input_reshaped_b_1_i)
# print(grad_for_raw_weight_before_sum_b_o_i, "grad_for_raw_weight_before_sum_b_o_i before sum")
# print(grad_for_raw_weight_before_sum_b_o_i.shape, "grad_for_raw_weight_before_sum_b_o_i.shape before sum")
# grad_for_raw_weight_o_i = grad_for_raw_weight_before_sum_b_o_i.sum(dim=0, keepdim=False)
# #grad_for_raw_weight_before_sum_b_o_i = grad_for_raw_weight_before_sum_b_o_i.squeeze(dim=0)
# print(grad_for_raw_weight_o_i, "grad_for_raw_weight_o_i after sum, the final result.")
# print(grad_for_raw_weight_o_i.shape, "grad_for_raw_weight_o_i.shape. Should be ", raw_weight.shape)

# one_hot_o_i = torch.zeros_like(raw_weight)
# one_hot_o_i[out_features_iota_o, index_of_max_o] = 1.
# one_hot_expanded_fake_b_o_i = one_hot_o_i.expand(g_in.shape[0], -1, -1)
# print(one_hot_expanded_fake_b_o_i.shape, "one_hot_expanded_fake_b_o_i.shape")
# print(one_hot_expanded_fake_b_o_i, "one_hot_expanded_fake_b_o_i")

# g_in_reshaped_expanded_b_o_fake_i = g_in_reshaped_b_o_1.expand(-1, -1, in_features_s)
# print(g_in_reshaped_expanded_b_o_fake_i, "g_in_reshaped_expanded_b_o_fake_i")
# print(g_in_reshaped_expanded_b_o_fake_i.shape, "g_in_reshaped_expanded_b_o_fake_i.shape")

# one_hot_mul_g_in_b_o_i = one_hot_expanded_fake_b_o_i.mul(g_in_reshaped_expanded_b_o_fake_i)
# print(one_hot_mul_g_in_b_o_i, "one_hot_mul_g_in_b_o_i")

# grad_for_x_b_i = one_hot_mul_g_in_b_o_i.sum(dim=1, keepdim=False)
# print(grad_for_x_b_i, "grad_for_x_b_i")
# print(grad_for_x_b_i.shape, "grad_for_x_b_i.shape")

# print("--------------SUMMARIZE-------------")
# print(output, "output")
# print(grad_for_x_b_i, "grad_for_x")
# print(grad_for_raw_weight_o_i, "grad_for_raw_weight")

# print("--------------VALIDATION-------------")

# __proxy_weight = torch.zeros_like(raw_weight)
# __proxy_weight.requires_grad_(True)
# __proxy_weight.data[out_features_iota_o, index_of_max_o] = 1.
# #print(__proxy_weight, "__proxy_weight")
# #print(__proxy_weight.shape, "__proxy_weight.shape")

# __valid_input_reshaped = x.unsqueeze(dim=-1)
# __valid_input_reshaped.requires_grad_(True)
# #print(__valid_input_reshaped, "__valid_input_reshaped")
# #print(__valid_input_reshaped.shape, "__valid_input_reshaped.shape")

# __valid_output = __proxy_weight.matmul(__valid_input_reshaped)
# __valid_output = __valid_output.squeeze(dim=-1)

# print(__valid_output, "__valid_output")
# #print(__valid_output.shape, "__valid_output.shape")

# #print(__valid_input_reshaped.grad, "__valid_input_reshaped.grad before")
# #print(__proxy_weight.grad, "__proxy_weight.grad before")
# torch.autograd.backward(__valid_output, g_in, inputs=[__valid_input_reshaped, __proxy_weight])
# print(__valid_input_reshaped.grad, "__valid_input_reshaped.grad after")
# print(__proxy_weight.grad, "__proxy_weight.grad after")

# print("-----------now test for the autograd function-----------")
# function_test__input = x.detach().clone().requires_grad_(True)
# function_test__input.grad = None
# function_test__raw_weight = raw_weight.detach().clone().requires_grad_(True)
# function_test__raw_weight.grad = None
# function_test__output = DigitalMapperFunction_v2.apply(function_test__input, function_test__raw_weight)
# print(function_test__output, "function_test__output")
# print(function_test__output.shape, "function_test__output.shape")
# torch.autograd.backward(function_test__output, g_in, inputs = \
#     [function_test__input, function_test__raw_weight])
# print(function_test__input.grad, "function_test__input.grad")
# print(function_test__raw_weight.grad, "function_test__raw_weight.grad")
# fds=432




class DigitalMapper_eval_only_v2(torch.nn.Module):
    r'''This class is not designed to be created by user directly. 
    Use DigitalMapper.can_convert_into_eval_only_mode 
    and DigitalMapper.convert_into_eval_only_mode to create this layer.
   
    And, if I only provide result in this form, it's possible to make puzzles.
    To solve the puzzle, figure the source of this layer.
    '''
    def __init__(self, in_features: int, out_features: int, indexes:torch.Tensor, \
                    device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if len(indexes.shape)!=1:
            raise Exception("Param:indexes must be a rank-1 tensor. This class is not designed to be created by user directly. Use DigitalMapper.can_convert_into_eval_only_mode and DigitalMapper.convert_into_eval_only_mode to create this layer.")
        self.indexes = torch.nn.Parameter(indexes, requires_grad=False)
        self.indexes.requires_grad_(False)
    pass
    def accepts_non_standard_range(self)->bool:
        return False
    def outputs_standard_range(self)->bool:
        return True    
    def outputs_non_standard_range(self)->bool:
        return not self.outputs_standard_range()
    def forward(self, input:torch.Tensor):
        x = input[:, self.indexes]
        #print(x)
        return x


class DigitalMapper_V2(torch.nn.Module):
    r'''This layer is NOT designed to be used directly.
    Use the wrapper class DigitalMapper.
    '''
    #__constants__ = []
    
    protect_param_every____training:int
    training_count:int
    
    def __init__(self, in_features: int, out_features: int, \
                        scaling_ratio_for_gramo:float = 20000., \
                        protect_param_every____training:int = 20, \
                        auto_print_difference:bool = False, \
                    device=None, dtype=None) -> None: 
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.raw_weight = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.reset_parameters()
        self.gramo = GradientModification(scaling_ratio=scaling_ratio_for_gramo)
        #self.auto_print_difference = auto_print_difference
        
        #self.raw_weight_before = torch.nn.param
        self.set_auto_print_difference_between_epochs(auto_print_difference)
        # the old way.
        #     self.raw_weight_before = None
        # else:
        #     self.raw_weight_before = torch.nn.Parameter(torch.empty_like(self.raw_weight), requires_grad=False)
        #     self.raw_weight_before.requires_grad_(False)
        #     self.raw_weight_before.data = self.raw_weight.detach().clone()
        #     pass
                
        # For the param protection.
        self.protect_param_every____training = protect_param_every____training 
        self.training_count = 0 
        self.threshold1 = torch.nn.Parameter(torch.tensor([3.]), requires_grad=False)
        self.threshold2 = torch.nn.Parameter(torch.tensor([7.]), requires_grad=False)
        self.margin = torch.nn.Parameter(torch.tensor([1.]), requires_grad=False)
        self.epi = torch.nn.Parameter(torch.tensor([0.01]), requires_grad=False)
        pass

    def reset_parameters(self) -> None:
        '''copied from torch.nn.Linear'''
        torch.nn.init.kaiming_uniform_(self.raw_weight, a=math.sqrt(5))
        pass
        
    def accepts_non_standard_range(self)->str:
        return "although this layer accepts non standard input, I recommend you only feed standard +-1(np) as input."
    def outputs_standard_range(self)->str:
        return "It depends on what you feed in. If the input is standard +-1(np), the output is also standard +-1(np)."    
    # def outputs_non_standard_range(self)->bool:
    #     return not self.outputs_standard_range()
    
    def set_scaling_ratio(self, scaling_ratio:float):
        '''simply sets the inner'''
        self.gramo.set_scaling_ratio(scaling_ratio)
        pass
    def get_scaling_ratio(self)->torch.nn.parameter.Parameter:
        '''simply gets the inner'''
        return self.gramo.scaling_ratio
    
    def set_param_protection_config(self, threshold1:float = -1., \
            threshold2:float = -1., margin:float = -1., epi:float = -1.):
        r'''Use any number <= 0. not to touch this param.
        
        example:
        >>> set_param_protection_config(-1., -1., 0.5, -1.)
        to set only margin.
        '''
        if threshold1<=0.:
            threshold1 = self.threshold1.item()
            pass
        if threshold2<=0.:
            threshold2 = self.threshold2.item()
            pass
        if margin<=0.:
            margin = self.margin.item()
            pass
        if epi<=0.:
            epi = self.epi.item()
            pass
        if threshold1+margin>=threshold2:
            raise Exception("threshold1+margin must be less than threshold2.")
        if epi>=threshold1 or epi>=threshold2 or epi>=margin:
            raise Exception("epi seems too big. If you know what you are doing, comment this line out.")
        
        self.threshold1 = torch.nn.Parameter(torch.tensor([threshold1]), requires_grad=False)
        self.threshold2 = torch.nn.Parameter(torch.tensor([threshold2]), requires_grad=False)
        self.margin = torch.nn.Parameter(torch.tensor([margin]), requires_grad=False)
        self.epi = torch.nn.Parameter(torch.tensor([epi]), requires_grad=False)
        raise Exception("untested.")
        pass
    
    # def __update_fake_weight(self):
    #     temp = self.raw_weight.data.max(dim=1, keepdim=False)
        
    #     self.fake_weight.data = torch.zeros_like(self.fake_weight.data)
    #     self.fake_weight[:][temp[:]] =1.
        
    #     self.fake_weight = self.fake_weight.requires_grad_()
            
    def set_auto_print_difference_between_epochs(self, set_to:bool = True):
        if not set_to:
            self.raw_weight_before = torch.nn.Parameter(torch.empty([0,]))
            # use self.raw_weight_before.nelement() == 0 to test it.
            pass
        if set_to:
            if self.raw_weight is None:
                raise Exception("This needs self.raw_weight first. Report this bug to the author, thanks.")
            if self.raw_weight.nelement() == 0:
                raise Exception("Unreachable code. self.raw_weight contains 0 element. It's so wrong.")
            #if not hasattr(self, "raw_weight_before") or self.raw_weight_before is None:
            if self.raw_weight_before is None:
                self.raw_weight_before = torch.nn.Parameter(torch.empty_like(self.raw_weight), requires_grad=False)
                self.raw_weight_before.requires_grad_(False)
                pass
            self.raw_weight_before.data = self.raw_weight.detach().clone()
            pass
        pass
            
    def get_index_format(self)->torch.Tensor:
        index_of_max_o = self.raw_weight.max(dim=1).indices
        return index_of_max_o
    def get_one_hot_format(self)->torch.Tensor:
        #raw_weight = torch.tensor([[1., 2., 3.], [4., 2., 3.], [4., 5., 8.], [6., 2., 9.],[6., 2., 9.], ])
        out_features_s = self.raw_weight.shape[0]
        out_features_iota_o = torch.linspace(0, out_features_s-1, out_features_s, dtype=torch.int32)
        #print(out_features_iota, "out_features_iota")
        index_of_max_o = self.raw_weight.max(dim=1).indices
        #print(index_of_max_o, "index_of_max_o")
        
        one_hot_o_i = torch.zeros_like(self.raw_weight)
        one_hot_o_i[out_features_iota_o, index_of_max_o] = 1.
        return one_hot_o_i
    
    def forward(self, input:torch.Tensor)->torch.Tensor:
        # If you know how pytorch works, you can comment this checking out.
        # if self.training and (not input.requires_grad):
        #     raise Exception("Set input.requires_grad to True. If you know what you are doing, you can comment this line.")
        
        # auto print difference between epochs        
        #if (hasattr(self, "raw_weight_before")): 
        
        
        # use self.raw_weight_before.nelement() == 0 to test it.
        if self.raw_weight_before.nelement() != 0:
            ne_flag = self.raw_weight_before.data.ne(self.raw_weight)
            if ne_flag.any()>0:
                to_report_from = self.raw_weight_before[ne_flag]
                to_report_from = to_report_from[:16]
                to_report_to = self.raw_weight[ne_flag]
                to_report_to = to_report_to[:16]
                line_number_info = "    Line number: "+str(sys._getframe(1).f_lineno)
                print("Raw weight changed, from:\n", to_report_from, ">>>to>>>\n", 
                        to_report_to, line_number_info)
            else:
                print("Raw weight was not changed in the last stepping")
                pass
            self.raw_weight_before.data = self.raw_weight.detach().clone()
            pass
            #pass
        
        if len(input.shape)!=2:
            raise Exception("DigitalMapper only accept rank-2 tensor. The shape should be[batch, input dim]")
        
        #param protection
        if self.training_count<self.protect_param_every____training:
            self.training_count +=1
        else:
            self.training_count = 0
            #param protection
            # the positive part
            target_exceed = self.threshold2-self.threshold1-self.margin
            gt_t2 = self.raw_weight.gt(self.threshold2)
            at_least_smt_gt_t2 = gt_t2.any(dim=1)
            if at_least_smt_gt_t2.any().item():
                the_max_value = self.raw_weight.max(dim=1, keepdim=True).values
                gt_t1 = self.raw_weight.gt(self.threshold1)
                at_least_smt_gt_t2_expanded = at_least_smt_gt_t2[:, None].expand(-1, self.raw_weight.shape[1])
                modify_these = gt_t1.logical_and(at_least_smt_gt_t2_expanded)
                exceed = the_max_value-self.threshold1
                exceed = exceed.abs()+self.epi#or exceed.mul(at_least_smt_gt_t2)+epi
                mul_me = target_exceed/exceed
                self.raw_weight.data = modify_these*((self.raw_weight-self.threshold1)*mul_me+self.threshold1)+(modify_these.logical_not())*self.raw_weight
            pass
            # the negative part
            lt_nt2 = self.raw_weight.lt(-1.*self.threshold2)
            at_least_smt_lt_nt2 = lt_nt2.any(dim=1)
            if at_least_smt_lt_nt2.any().item():
                the_min_value = self.raw_weight.min(dim=1, keepdim=True).values
                lt_nt1 = self.raw_weight.lt(-1.*self.threshold1)
                at_least_smt_lt_nt2_expanded = at_least_smt_lt_nt2[:, None].expand(-1, self.raw_weight.shape[1])
                modify_these_negative = lt_nt1.logical_and(at_least_smt_lt_nt2_expanded)
                exceed = the_min_value+self.threshold1
                exceed = exceed.abs()+self.epi#or exceed.mul(at_least_smt_gt_t2)+epi
                mul_me_negative = target_exceed/exceed
                self.raw_weight.data = modify_these_negative*((self.raw_weight+self.threshold1)*mul_me_negative-self.threshold1) \
                    +(modify_these_negative.logical_not())*self.raw_weight
            pass
        # end of param protection.
        
        if self.training:
            w = self.gramo(self.raw_weight)
            x = DigitalMapperFunction_v2.apply(input, w)
            return x
        else:#eval mode:
            w = self.raw_weight
            x = DigitalMapperFunction_v2.apply(input, w)
            return x
        
   
    def get_eval_only(self)->DigitalMapper_eval_only_v2:
        return DigitalMapper_eval_only_v2(self.in_features, self.out_features,self.get_index_format())
   
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}'
    
    pass


# '''Param protection test. Real case vs equivalent individual code.'''
# layer = DigitalMapper_V2(3,8,protect_param_every____training=0)
# layer.raw_weight.data = torch.tensor([
#     [1., 2., 3.],[1., 3., 6.],[3., 6., 11.],[5., 10., 15.],
#     [-10., 2., 10.],[-1., -3., -6.],[-3., -6., -11.],[-5., -10., -15.],
#     ])
# layer(torch.tensor([[1., 2., 3.]], requires_grad=True))
# print(layer.raw_weight)
# fds=432
# '''Then ,the individual code.'''
# epi = 0.01
# threshold1 = 3.
# threshold2 = 7.
# margin = 1.

# target_exceed = threshold2-threshold1-margin
# test_weight = torch.tensor([
#     [1., 2., 3.],[1., 3., 6.],[3., 6., 11.],[5., 10., 15.],
#     [-10., 2., 10.],[-1., -3., -6.],[-3., -6., -11.],[-5., -10., -15.],
#     ])
# gt_t2 = test_weight.gt(threshold2)
# #print(gt_t2, "gt_t2")
# at_least_smt_gt_t2 = gt_t2.any(dim=1)
# #print(at_least_smt_gt_t2, "at_least_smt_gt_t2")
# if at_least_smt_gt_t2.any().item():
#     the_max_value = test_weight.max(dim=1, keepdim=True).values
#     #print(the_max_value, "the_max_value")
#     gt_t1 = test_weight.gt(threshold1)
#     #print(gt_t1, "gt_t1")
    
#     at_least_smt_gt_t2_expanded = at_least_smt_gt_t2[:, None].expand(-1, test_weight.shape[1])
#     #print(at_least_smt_gt_t2_expanded.shape, "at_least_smt_gt_t2_expanded.shape")
    
#     modify_these = gt_t1.logical_and(at_least_smt_gt_t2_expanded)
#     #print(modify_these, "modify_these")
    
#     exceed = the_max_value-threshold1
#     exceed = exceed.abs()+epi#or exceed.mul(at_least_smt_gt_t2)+epi
#     #print(exceed, "exceed")
    
#     mul_me = target_exceed/exceed
#     print(mul_me, "mul_me")
#     test_weight = modify_these*((test_weight-threshold1)*mul_me+threshold1)+(modify_these.logical_not())*test_weight
#     print(test_weight, "test_weight")
# pass
    
# #test_weight2 = test_weight.detach().clone()*-1
# #print(test_weight2, "test_weight2")
# lt_nt2 = test_weight.lt(-1.*threshold2)
# #print(lt_nt2, "lt_nt2")
# at_least_smt_lt_nt2 = lt_nt2.any(dim=1)
# #print(at_least_smt_lt_nt2, "at_least_smt_lt_nt2")
# if at_least_smt_lt_nt2.any().item():
#     the_min_value = test_weight.min(dim=1, keepdim=True).values
#     #print(the_min_value, "the_min_value")
#     lt_nt1 = test_weight.lt(-threshold1)
#     #print(lt_nt1, "lt_nt1")
    
#     at_least_smt_lt_nt2_expanded = at_least_smt_lt_nt2[:, None].expand(-1, test_weight.shape[1])
#     #print(at_least_smt_lt_nt2_expanded, "at_least_smt_lt_nt2_expanded")
#     #print(at_least_smt_lt_nt2_expanded.shape, "at_least_smt_lt_nt2_expanded.shape")
    
#     modify_these_negative = lt_nt1.logical_and(at_least_smt_lt_nt2_expanded)
#     #print(modify_these_negative, "modify_these_negative")
    
#     exceed = the_min_value+threshold1
#     exceed = exceed.abs()+epi#or exceed.mul(at_least_smt_gt_t2)+epi
#     #print(exceed, "exceed")
    
#     mul_me_negative = target_exceed/exceed
#     print(mul_me_negative, "mul_me_negative")
#     test_weight = modify_these_negative*((test_weight+threshold1)*mul_me_negative-threshold1) \
#         +(modify_these_negative.logical_not())*test_weight
#     print(test_weight, "test_weight")
# pass
# print("--------------SUMMARIZATION-----------")
# print(layer.raw_weight, "real code.")
# print(test_weight, "test code.")
# fds=432



# '''Test for all the modes, and eval only layer.'''
# print("All 4 prints below should be equal.")
# x = torch.tensor([[5., 6., 7.], [8., 9., 13]], requires_grad=True)
# layer = DigitalMapper_V2(3,5)
# print(layer(x))
# print(x[:,layer.get_index_format()])
# print(layer.get_one_hot_format().matmul(x[:,:,None]).squeeze(dim=-1))
# fds=432
# eval_only_layer = layer.get_eval_only()
# print(eval_only_layer(x))
# print("All 4 prints above should be equal.")

# fds=432


# '''basic test. Also, the eval mode.'''
# layer = DigitalMapper_V2(2,3)
# # print(layer.raw_weight.data.shape)
# layer.raw_weight.data=torch.Tensor([[2., -2.],[ 0.1, 0.],[ -2., 2.]])
# # print(layer.raw_weight.data)
# input = torch.tensor([[1., 0.]], requires_grad=True)
# print(layer(input), "should be 1 1 0")

# layer = DigitalMapper_V2(2,3)
# layer.eval()
# layer.raw_weight.data=torch.Tensor([[2., -2.],[ 0.1, 0.],[ -2., 2.]])
# input = torch.tensor([[1., 0.]], requires_grad=True)
# print(layer(input), "should be 1 1 0")
# fds=432



# '''some real training'''
# model = DigitalMapper_V2(2,1)
# model.set_scaling_ratio(200.)
# loss_function = torch.nn.MSELoss()
# input = torch.Tensor([[1., 1.],[1., 0.],[0., 1.],[0., 0.],])
# #input = torch.Tensor([[1., 0.],[0., 1.]])
# input = input.requires_grad_()
# target = torch.Tensor([[1.],[1.],[0.],[0.],])
# #target = torch.Tensor([[1.],[0.],])
# # print(input)
# # print(target)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
# # for p in model.parameters():
# #     print(p)

# iter_per_print = 3#1111
# print_count = 5
# for epoch in range(iter_per_print*print_count):
    
#     model.train()
#     pred = model(input)
    
#     # if epoch%iter_per_print == iter_per_print-1:
#     #     print(pred.shape, "pred")
#     #     print(target.shape, "target")
    
#     loss = loss_function(pred, target)
#     if True and epoch%iter_per_print == iter_per_print-1:
#         print(loss.item())
#         pass
#     optimizer.zero_grad()
#     loss.backward()
#     # if epoch%iter_per_print == iter_per_print-1:
#     #     print(model.raw_weight.grad, "grad")
        
#     #model.raw_weight.grad = model.raw_weight.grad*-1.
#     #optimizer.param_groups[0]["lr"] = 0.01
#     # if epoch%iter_per_print == iter_per_print-1:
#     #     print(model.raw_weight, model.raw_weight.grad, "before update")
#     # if epoch%iter_per_print == iter_per_print-1:
#     #     print(model.raw_weight, "raw weight itself")
#     #     print(model.raw_weight.grad, "grad")
#     #     fds=432
#     if True and epoch%iter_per_print == iter_per_print-1:
#         print(model.raw_weight)
#         optimizer.step()
#         print(model.raw_weight)
#         pass
    
#     optimizer.step()
#     # if epoch%iter_per_print == iter_per_print-1:
#     #     print(model.raw_weight, "after update")

#     model.eval()
#     if epoch%iter_per_print == iter_per_print-1:
#         # print(loss, "loss")
#         # if True:
#         #     print(model.raw_weight.softmax(dim=1), "eval after softmax")
#         #     print(model.raw_weight, "eval before softmax")
#         #     print("--------------")
#         pass

# model.eval()
# print(model(input), "should be", target)

# fds=432





'''Dry stack test. OK, the dry is actually a Chinese word, which means, only, or directly.'''
class test_directly_stacking_multiple_digital_mappers(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, mid_width: int, \
            auto_print_difference:bool = False, \
            layer_count = 2, device=None, dtype=None) -> None: 
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.mid_width = mid_width
        
        self.digital_mappers = torch.nn.ParameterList([])
        self.digital_mappers.append(DigitalMapper_V2(in_features,mid_width, 
                                        auto_print_difference = auto_print_difference))
        for _ in range(10):# I know what you are thinking. I know it.
            self.digital_mappers.append(DigitalMapper_V2(mid_width,mid_width, 
                                        auto_print_difference = auto_print_difference))
        self.digital_mappers.append(DigitalMapper_V2(mid_width,out_features, 
                                        auto_print_difference = auto_print_difference))
        pass
    def forward(self, input:torch.Tensor)->torch.Tensor:
        x = input
        for layer in self.digital_mappers:
            x = layer(x)
        return x
    pass

def data_gen_for_directly_stacking_test(batch:int, n_in:int, n_out:int, dtype = torch.float32)->Tuple[torch.Tensor, torch.Tensor]:
    input = torch.randint(0,2,[batch, n_in]).to(dtype)
    answer_index = torch.randint(0,n_in,[n_out])
    target = input[:, answer_index]
    raise Exception("untested.")
    return input, target


# '''some real training'''
# batch = 100
# n_in = 8
# n_out = 4
# model = test_directly_stacking_multiple_digital_mappers(n_in, n_out, 16, False)
# #model.set_scaling_ratio(200.)
# loss_function = torch.nn.MSELoss()
# if False:
#     '''this test passed. '''
#     input = torch.Tensor([[1., 1.],[1., 0.],[0., 1.],[0., 0.],])
#     target = torch.Tensor([[1.],[1.],[0.],[0.],])
#     pass
# if True:
#     input = torch.randint(0,2,[batch, n_in]).to(torch.float32)
#     answer_index = torch.randint(0,n_in,[n_out])
#     target = input[:, answer_index]
#     pass
# # print(input)
# # print(target)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
# '''0.0001 10%<100 epochs, <5000epochs'''
# '''0.001 60%<100 epochs. 3000epochs'''
# '''0.01 70%<100 epochs. <3000epochs'''

# iter_per_print = 111#1111
# print_count = 5555
# for epoch in range(iter_per_print*print_count):
#     model.train()
#     pred = model(input)
#     loss = loss_function(pred, target)
#     if True and epoch%iter_per_print == iter_per_print-1:
#         print(str(epoch+1), "     epochs/acc     ", bitwise_acc(pred, target))
#         pass
#     optimizer.zero_grad()
#     loss.backward()
#     if True and "Slightly Distortion for Grad" and epoch%iter_per_print == iter_per_print-1:
#         make_grad_noisy(model, 1.2)
#         pass
#     if False and epoch%iter_per_print == iter_per_print-1:
#         print("---------------------------------")
#         print(model.digital_mappers[1].raw_weight)
#         optimizer.step()
#         print(model.digital_mappers[1].raw_weight)
#         pass
#     optimizer.step()

# model.eval()
# #print(model(input), "should be", target)
# bitwise_acc(model(input), target, True)
# fds=432
















