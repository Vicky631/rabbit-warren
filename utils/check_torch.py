try:
    import torch
    print("torch库可用")
except ImportError:
    print("torch库不可用")
try:
    import torchvision
    print("torchvision库可用")
except ImportError:
    print("torchvision库不可用")