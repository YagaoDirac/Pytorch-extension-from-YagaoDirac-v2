import torch

def init_the_data()->torch.Tensor:
    shape = [2,3,5]
    n_element = shape[0]*shape[1]*shape[2]
    return torch.linspace(0,n_element-1,n_element).reshape(shape=shape)


"select"
a = init_the_data()
b = a.select(dim=2,index=1)
assert b.eq(torch.tensor([[   1.,  6., 11.],
                            [16., 21., 26.]])).all()
b[0,0] = 111
assert a[0,0,1] == 111#b is view.

"index_select"#creates a new tensor.
a = init_the_data()
b = a.index_select(dim=2,index=torch.tensor([1,3]))
assert b.eq(torch.tensor( [[[ 1.,  3.],
                            [ 6.,  8.],
                            [11., 13.]],

                            [[16., 18.],
                            [21., 23.],
                            [26., 28.]]])).all()
b[0,0,0] = 111
assert a[0,0,1] == 1#b is new tensor.





# torch.Tensor.split
# def split(
#     self: Tensor,
#     split_size: Any,
#     dim: int = 0
# ) -> (Any | tuple[Tensor, ...])

# torch.Tensor.split_with_sizes
# def split_with_sizes(
#     self: Tensor,
#     split_sizes: Sequence[int | SymInt],
#     dim: int = 0
# ) -> tuple[Tensor, ...]

# torch.Tensor.tensor_split
# def tensor_split(
#     self: Tensor,
#     tensor_indices_or_sections: Tensor,
#     dim: int = 0
# ) -> tuple[Tensor, ...]: ...

