import tensorflow as tf
import torch
import onnx



fname = 'models/theta_test.pt'

model = torch.load(fname)

torch.onnx.export(model, 
