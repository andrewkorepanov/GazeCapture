import torch

IMAGE_SIZE = (224, 224)

WORKERS = 8
EPOCHS = 25
BATCH_SIZE = torch.cuda.device_count() * 100  # Change if out of cuda memory

# LEARNING_RATE = 0.0001 #SGD
LEARNING_RATE = 0.002 #Adam
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4