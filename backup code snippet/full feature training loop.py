
(bit, layer, gates, batch, lr, noise_base) = test_config_dispatcher(
    bit=2,layer=6,gates=16,batch=100,lr=0.001,noise_base=1.2)
is_f16 = True
iter_per_print = 50#1111
print_count = 20000

(input, target) = data_gen_full_adder(bit,batch, is_output_01=False, is_cuda=True)
# print(input[:5])
# print(target[:5])

model = DSPU(input.shape[1],target.shape[1],gates*3,gates,gates,gates,layer, 
             scaling_ratio_for_gramo_in_mapper=1000.,
             scaling_ratio_for_gramo_in_mapper_for_first_layer=330.,
             scaling_ratio_for_gramo_in_mapper_for_out_mapper=1000.,)#1000,330,1000
model.get_info(directly_print=True)
#model.set_auto_print_difference_between_epochs(True,4)
#model.set_auto_print_difference_between_epochs(True,5)



model.cuda()
if is_f16:
    input = input.to(torch.float16)
    target = target.to(torch.float16)
    model.half()
    pass

# model_DSPU.print_mapper_scaling_ratio_for_inner_raw_from_all()
# model_DSPU.print_gates_big_number_from_all()
# fds=432


# print(model_DSPU.get_mapper_scaling_ratio_for_inner_raw_from_all())
# print(model_DSPU.get_gates_big_number_from_all())
#model_DSPU.first_layer.set_scaling_ratio()


#model.set_auto_print_difference_between_epochs(True,5)
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

if True and "print parameters":
    if True and "only the training params":
        for name, p in zip(model._parameters, model.parameters()):
            if p.requires_grad:
                print(name, p)
                pass
            pass
        pass
    else:# prints all the params.
        for name, p in zip(model._parameters, model.parameters()):
            print(name, p)
            pass
        pass

model.cuda()
input = input.cuda()
target = target.cuda()
#iter_per_print = 1#111
#print_count = 1555
for epoch in range(iter_per_print*print_count):
    model.train()
    pred = model(input)
    #print(pred, "pred", __line__str())
    if False and "shape":
        print(pred.shape, "pred.shape")
        print(target.shape, "target.shape")
        fds=423
    if False and "print pred":
        if epoch%iter_per_print == iter_per_print-1:
            print(pred[:5], "pred")
            print(target[:5], "target")
            pass
        pass
    loss = loss_function(pred, target)
    optimizer.zero_grad()
    loss.backward()
    if True and "make_grad_noisy":
        make_grad_noisy(model, noise_base)
        pass
    if False and "print the grad":
        if epoch%iter_per_print == iter_per_print-1:
            print(model.first_layer.in_mapper.raw_weight.grad, "first_layer   grad")
            print(model.second_to_last_layers[0].in_mapper.raw_weight.grad, "second_to_last_layers[0]   grad")
            print(model.out_mapper.raw_weight.grad, "out_mapper   grad")
            pass
        pass
    if False and "print the weight":
        if epoch%iter_per_print == iter_per_print-1:
            #layer = model.out_mapper
            layer = model.first_layer.in_mapper
            print(layer.raw_weight, "first_layer.in_mapper   before update")
            optimizer.step()
            print(layer.raw_weight, "first_layer.in_mapper   after update")
            
            layer = model.model.second_to_last_layers[0]
            print(layer.raw_weight, "second_to_last_layers[0]   before update")
            optimizer.step()
            print(layer.raw_weight, "second_to_last_layers[0]   after update")
            
            layer = model.out_mapper
            print(layer.raw_weight, "out_mapper   before update")
            optimizer.step()
            print(layer.raw_weight, "out_mapper   after update")
            
            pass    
        pass    
    if True and "print zero grad ratio":
        if epoch%iter_per_print == iter_per_print-1:
            result = model.get_zero_grad_ratio()
            print("print zero grad ratio: ", result)
            pass
        pass
    #optimizer.param_groups[0]["lr"] = 0.01
    optimizer.step()
    if True and "print param overlap":
        every = 100
        if epoch%every == every-1:
            model.print_param_overlap_ratio()
            pass
        pass
    if True and "print acc":
        if epoch%iter_per_print == iter_per_print-1:
            with torch.inference_mode():
                model.eval()
                pred = model(input)
                #print(pred, "pred", __line__str())
                #print(target, "target")
                acc = DigitalMapper_V1_1.bitwise_acc(pred, target)
                model.set_acc(acc)
                if 1. != acc:
                    print(epoch+1, "    ep/acc    ", acc)
                else:
                    #print(epoch+1, "    ep/acc    ", acc)
                    finished = model.can_convert_into_eval_only_mode()
                    print(finished, "is param hard enough __line 1273")
                    if finished[0]:
                        print(pred[:5].T, "pred", __line__str())
                        print(target[:5].T, "target")
                        break
                        pass
                    pass
                pass
            pass
        pass

fds=432