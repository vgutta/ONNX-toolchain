import json
idx2label = []
cls2label = {}
with open("imagenet_class_index.json", "r") as read_file:
    class_idx = json.load(read_file)
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    cls2label = {class_idx[str(k)][0]: class_idx[str(k)][1] for k in range(len(class_idx))}

from torchvision import models
import torch
from torchvision import transforms

dummy_input = torch.randn(1, 3, 227, 227)
squeezenet = models.squeezenet1_1(pretrained=True) ## downloads pretrained model

input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
output_names = [ "output1" ]

torch.onnx.export(squeezenet, dummy_input, "../output/squeezenet.onnx", verbose=True, input_names=input_names, output_names=output_names)

transform = transforms.Compose([            #[1]
 transforms.Resize(256),                    #[2]
 transforms.CenterCrop(224),                #[3]
 transforms.ToTensor(),                     #[4]
 transforms.Normalize(                      #[5]
 mean=[0.485, 0.456, 0.406],                #[6]
 std=[0.229, 0.224, 0.225]                  #[7]
 )])

from PIL import Image
img = Image.open("dog.jpg")
img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0)
squeezenet.eval()
out = squeezenet(batch_t)

_, index = torch.max(out, 1)
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
print(idx2label[index[0]], percentage[index[0]].item())

import onnxruntime
from onnxruntime import InferenceSession, SessionOptions, get_all_providers

EXECUTION_PROVIDER = "CUDAExecutionProvider"

img = Image.open("dog.jpg")
resize = transforms.Resize([227, 227])
img_t = resize(img)
to_tensor = transforms.ToTensor()
img_y = to_tensor(img_t)
img_y.unsqueeze_(0)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def create_model_for_provider(model_path: str, provider: str) -> InferenceSession: 
  
  assert provider in get_all_providers(), f"provider {provider} not found, {get_all_providers()}"

  # Few properties than might have an impact on performances (provided by MS)
  options = SessionOptions()
  options.intra_op_num_threads = 1

  # Load the model as a graph and prepare the CPU backend 
  return InferenceSession(model_path, options, providers=[provider])

cpu_model = create_model_for_provider("../output/squeezenet.onnx", EXECUTION_PROVIDER)

ort_inputs = {cpu_model.get_inputs()[0].name: to_numpy(img_y)}

## Here first arguments None 
# becuase we want every output sometimes model return more than one output
output = cpu_model.run(None, ort_inputs) 

from timeit import timeit

print("PyTorch SqueezeNet")
print(timeit("squeezenet(batch_t)", number=100, globals=globals()))

print("ONNX Runtime SqueezeNet CPUExecutionProvider")
print(timeit("cpu_model.run(None, ort_inputs)", number=100, globals=globals()))