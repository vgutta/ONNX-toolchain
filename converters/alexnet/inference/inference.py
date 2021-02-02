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

alexnet = models.alexnet(pretrained=True)

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
alexnet.eval()
out = alexnet(batch_t)

_, index = torch.max(out, 1)
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
print(idx2label[index[0]], percentage[index[0]].item())

torch.onnx.export(
    alexnet, ## pass model
    (batch_t), ## pass inpout example
    "../alexnet.onnx", ##output path
    input_names = ['input'], ## Pass names as per model input name
    output_names = ['output'], ## Pass names as per model output name
    opset_version=12, ##  export the model to the  opset version of the onnx submodule.
    dynamic_axes = { ## this will makes export more generalize to take batch for prediction
        'input' : {0: 'batch', 1: 'sequence'},
        'output' : {0: 'batch'},
    }
)
