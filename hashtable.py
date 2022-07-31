import torch
import random
import numpy as np
nums  = torch.arange(10)
stuff = [2,4,6,8,10,12,14,16,18,20]
choice = np.random.choice(stuff, 5, replace=False)

print(choice)
