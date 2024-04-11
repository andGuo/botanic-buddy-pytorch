import os
import torch
torch.cuda.empty_cache()
from ultralytics import YOLO

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model

# Use the model
model.train(data="data.yaml", epochs=100, batch=4)  # train the model
metrics = model.val()  # evaluate model performance on the validation set

path = model.export(format="coreml", nms=True)
