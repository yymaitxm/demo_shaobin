import json

import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import glob
from resnet_pytorch import ResNet 

# Open image
imglists = glob.glob("/Users/yiming.yin/PycharmProjects/pythonProject/demo_data/pic/*.png")
#imglists.sort()

for imgname in imglists:
    imgname2 = int(imgname.split("/")[-1].split(".")[0]) + 30
    imgname2 = "/Users/mingming.ma/Documents/data/哨兵/pic/" + str(imgname2).zfill(5) + ".png"
    input_image = Image.open(imgname)
    input_image2 = Image.open(imgname2)
    input_image = input_image.crop((256*2,0, 256 * 3, 200))
    input_image2 = input_image2.crop((256*2,0, 256 *3, 200))

    # Preprocess image
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
    input_tensor2 = preprocess(input_image2)
    input_batch2 = input_tensor2.unsqueeze(0)

    # Load class names
    # labels_map = json.load(open("labels_map.txt"))
    # labels_map = [labels_map[str(i)] for i in range(1000)]

    # Classify with ResNet18
    model = ResNet.from_pretrained("resnet50")
    model.eval()
    with torch.no_grad():
        logits = model.extract_features(input_batch)
        logits = logits.mean(dim=1)
        logits2 = model.extract_features(input_batch2)
        logits2 = logits2.mean(dim=1)
        diff_fe = abs(logits2 - logits)
        print("diff max: ", diff_fe.max())
        print("diff min: ", diff_fe.min())
        diff_img = diff_fe.numpy().squeeze(0)
        diff_num = np.where(diff_img > 0.05)
        diff_img_re = cv2.resize(diff_img, (256, 200)) * 15
        image = np.concatenate((input_image, input_image2),axis=1)
        cv2.imshow("img", image)
        cv2.imshow("fea", diff_img_re)
        cv2.waitKey(0)
        print("diff_fe", diff_fe)
    print("-----")

    # move the input and model to GPU for speed if available
    # if torch.cuda.is_available():
    #     input_batch = input_batch.to("cuda")
    #     model.to("cuda")




# preds = torch.topk(logits, k=5).indices.squeeze(0).tolist()

# print("top5: ", preds)
# print("top5: ", torch.topk(logits, k=5))

# for idx in preds:
#     label = labels_map[idx]
#     prob = torch.softmax(logits, dim=1)[0, idx].item()
#     print(f"{label:<75} ({prob * 100:.2f}%)")