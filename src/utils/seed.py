import random, os, numpy as np, torch

def set_seed(sd: int = 42):
    random.seed(sd)
    np.random.seed(sd)
    os.environ["PYTHONHASHSEED"] = str(sd)
    torch.manual_seed(sd)
    torch.cuda.manual_seed_all(sd)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False