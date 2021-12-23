import json

import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
from resnet_pytorch import ResNet

import glob

# Open image random sort 3513
imglists = glob.glob("/Users/yiming.yin/PycharmProjects/pythonProject/demo_data/pic/*.png")

# imglists.sort()


for imgname in imglists:
    # 这个地方为什么要将图片序号 + 30？
    imgname2 = int(imgname.split("/")[-1].split(".")[0]) + 30
    # 填充0至指定宽度
    imgname2 = "/Users/yiming.yin/PycharmProjects/pythonProject/demo_data/pic/" + str(imgname2).zfill(5) + ".png"

    input_image = Image.open(imgname)
    input_image2 = Image.open(imgname2)
    # 左上右下，为什么是256呢不应该是250吗
    input_image = input_image.crop((256*2,0, 256 * 3, 200))
    input_image2 = input_image2.crop((256*2,0, 256 *3, 200))

    # Preprocess image
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # 3*224*224
    input_tensor = preprocess(input_image)
    print(input_tensor.size())
    # 1*3*224*224
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
    print(input_batch.size())

    input_tensor2 = preprocess(input_image2)
    input_batch2 = input_tensor2.unsqueeze(0)

    # Load class names
    # labels_map = json.load(open("labels_map.txt"))
    # labels_map = [labels_map[str(i)] for i in range(1000)]

    # Classify with ResNet50
    model = ResNet.from_pretrained("resnet50")
    model.eval()
    with torch.no_grad():
        #  """ Returns output of the final convolution layer """
        logits = model.extract_features(input_batch)
        print(logits.size())

        logits = logits.mean(dim=1)
        print(logits.size())

        logits2 = model.extract_features(input_batch2)
        logits2 = logits2.mean(dim=1)

        diff_fe = abs(logits2 - logits)
        print("diff max: ", diff_fe.max())
        print("diff min: ", diff_fe.min())

        # print(type(diff_fe))

        diff_img = diff_fe.numpy().squeeze(0)

        print(type(diff_img))
        # 0.05是如何得出来的呀？避免微弱的灯光影响吗？
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
