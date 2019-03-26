from torch.utils.data import Dataset

def shuffle(a,b):
    import random
    start_state = random.getstate()
    random.shuffle(a)
    random.setstate(start_state)
    random.shuffle(b)

# if __name__ == '__main__':
#     a = list(range(1,10))
#     b = list(range(1,10))
#     for i in range(10):
#         shuffle(a,b)
#         print(a)
#         print(b)

import torch

# x= torch.rand(4,5)

x=torch.Tensor([21,42,45,59])

print(x)

index=torch.tensor([1,2,0,3])

x=x.index_copy_(0,index,x)

print(x)
