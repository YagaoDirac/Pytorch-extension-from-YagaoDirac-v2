import torch

short_tensor = torch.tensor([1.,0.3, -2], requires_grad=True)
result_of_short = short_tensor.softmax(dim=0)
print("result_of_short")
print(result_of_short)

result_of_short.backward(gradient=torch.tensor([1.,2,3]), inputs=[short_tensor])
print("short_tensor.grad")
print(short_tensor.grad)
print()


if "wrong plain version" and False:
    long_tensor = torch.tensor([1.,0.3, -2, 0,0,0,0,0])
    result_of_plain_long = long_tensor.softmax(dim=0)
    print("result_of_plain_long")
    print(result_of_plain_long)

    result_of_plain_long.backward(gradient=torch.tensor([1.,2,3,  123,432,345,-654,-567]), inputs=[long_tensor])
    #result_of_plain_long.backward(gradient=torch.tensor([1.,2,3, 0,0,0,0,0]), inputs=[long_tensor])
    print("result_of_plain_long.grad")
    print(result_of_plain_long.grad)
    print()
    pass


if "hard cut" and False:
    long_tensor_for_test = torch.tensor([1.,0.3, -2, 0,0,0,0,0], requires_grad=True)
    flag_non_zero___d = long_tensor_for_test.ne(0.)#this step kills grad. But it's possible to make a soft ver?
    the_exp___d = (long_tensor_for_test-long_tensor_for_test.max()).exp()# safe softmax.
    denominator = (the_exp___d*flag_non_zero___d).sum()
    new_version___d = the_exp___d/denominator * flag_non_zero___d
    print("new_version___d")
    print(new_version___d)

    new_version___d.backward(gradient=torch.tensor([1.,2,3,  123,432,345,-654,-567]), inputs=[long_tensor_for_test])
    print("long_tensor_for_test.grad")
    print(long_tensor_for_test.grad)
    print()
    pass


if "soft cut" and True:
    long_tensor_for_test = torch.tensor([1.,0.3, -2, 0,0,0,0,0], requires_grad=True)
    flag_non_zero___d = long_tensor_for_test.abs().pow(0.3).sub(0.1).mul(5.).clamp(0.,1.)
    the_exp___d = (long_tensor_for_test-long_tensor_for_test.max()).exp()# safe softmax.
    flagged_exp___d = the_exp___d* flag_non_zero___d
    denominator = flagged_exp___d.sum()
    new_version___d = flagged_exp___d/denominator
    print("new_version___d")
    print(new_version___d)

    #new_version___d.backward(gradient=torch.tensor([1.,2,3,  123,432,345,-654,-567]), inputs=[long_tensor_for_test])
    new_version___d.backward(gradient=torch.tensor([1.,2,3, 0,0,0,0,0]), inputs=[long_tensor_for_test])
    print("long_tensor_for_test.grad")
    print(long_tensor_for_test.grad)
    print()
    pass



