import torch

dtype_list:list[torch.dtype] = [torch.bool, 
            torch.int, torch.int8, torch.int16, torch.int32, torch.int64, 
            torch.uint8, torch.uint16, torch.uint32, torch.uint64,
            torch.float, torch.float16, torch.float32, torch.float64, 
            torch.bfloat16, 
            torch.complex64, torch.complex128, ]

for dtype in dtype_list:

    a = torch.empty(size=[1], dtype=dtype)
    print(f"{str(dtype):16}  {"fp" if a.is_floating_point() else "  " \
            } {"complex" if a.is_complex() else "       " \
            } {"quantized" if a.is_quantized else "         " \
            } ")

#then you get
#conclusion, no function to tell you if something is int/uint.
# torch.bool                             
# torch.int32                            
# torch.int8                             
# torch.int16                            
# torch.int32                            
# torch.int64                            
# torch.uint8                            
# torch.uint16                           
# torch.uint32                           
# torch.uint64                           
# torch.float32     fp                   
# torch.float16     fp                   
# torch.float32     fp                   
# torch.float64     fp                   
# torch.bfloat16    fp                   
# torch.complex64      complex           
# torch.complex128     complex    