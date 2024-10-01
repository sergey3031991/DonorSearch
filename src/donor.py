import io

from PIL import Image
import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.transforms import v2


CLASSES = {"0": 0, "90": 1, "180": 2, "270": 3}
IMG_SIZE = 255
transformer = v2.Compose(
    [
        v2.Pad(padding=10, fill=0, padding_mode="constant"),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(255),
        v2.CenterCrop((IMG_SIZE, IMG_SIZE)),
    ]
)

pretrained_weights = torch.load(
    "resnet_trained.pth", map_location=torch.device("cpu")
)
model_trained = resnet50(pretrained=False)

# Set requires_grad = False for all parameters
for param in model_trained.parameters():
    param.requires_grad = False

# Re-enable requires_grad for the last 3 layers
layers = list(model_trained.children())
for layer in layers[-4:]:
    for param in layer.parameters():
        param.requires_grad = True
in_features = 2048
out_features = 4
model_trained.fc = nn.Linear(in_features, out_features)
model_trained.load_state_dict(pretrained_weights)
model_trained.eval()


def predict_rotate(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    input = transformer(img)
    output = model_trained(input.unsqueeze(0))
    prediction = int(torch.max(output.cpu().data, 1)[1].numpy())
    if prediction != 0:
        classes_list = list(CLASSES.keys())
        rotation_angle = 360 - int(classes_list[prediction])
        rot_img = img.rotate(rotation_angle, expand=True)
        img_byte_arr = io.BytesIO()
        rot_img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        return img_byte_arr
    return io.BytesIO(img_bytes)
