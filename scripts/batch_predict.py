import os
import time
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from models.segformer.segformer import SegFormer


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def batch_predict(imgs_root, img_save_path):
    num_classes = 1 + 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # dcd
    mean = (0.570, 0.564, 0.555)
    std = (0.102, 0.102, 0.102)

    data_transform = transforms.Compose(
        [transforms.Resize(512),
         transforms.ToTensor(),
         transforms.Normalize(mean=mean, std=std)])

    images_list = os.listdir(imgs_root)

    # create model
    model = SegFormer(num_classes=num_classes, phi="b0")

    # load model weights
    weights_path = "trained_weights/segformer/20230407-024559-best_model.pth"  # dcd
    assert os.path.exists(weights_path), f"file: '{weights_path}' dose not exist."

    pretrain_weights = torch.load(weights_path, map_location='cpu')
    if "model" in pretrain_weights:
        model.load_state_dict(pretrain_weights["model"])
    else:
        model.load_state_dict(pretrain_weights)
    model.to(device)

    # prediction
    model.eval()
    with torch.no_grad():
        for index, image in enumerate(images_list):
            original_img = Image.open(os.path.join(imgs_root, image)).convert("RGB")
            img = data_transform(original_img)
            img = torch.unsqueeze(img, dim=0)

            output = model(img.to(device))
            prediction = output['out'].argmax(1).squeeze(0)
            prediction = prediction.to("cpu").numpy().astype(np.uint8)
            prediction[prediction == 0] = 255
            prediction[prediction == 1] = 0
            mask = Image.fromarray(prediction)
            mask.save(os.path.join(img_save_path, image))

            print("\r[{}] detecting [{}/{}]".format(image, index + 1, len(images_list)), end="")  # processing bar
        print()
