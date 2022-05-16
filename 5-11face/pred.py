import torch
import numpy as np
from model import AlexNet
from torchvision import transforms
import os
from PIL import Image
import json
import matplotlib.pyplot as plt


def pred():
    #设备
    device = torch.device("cpu")
    #模型定义
    model = AlexNet(num_classes=args.num_classes)
    model = model.to(device)
    # 可以显示汉字
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    #加载模型权重
    model.load_state_dict(torch.load(args.model_pth,map_location=device))
    data_path = args.data_folder
    #数据处理
    transform = transforms.Compose([transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    f = open("./classes.json",'r',encoding='utf-8')
    cls_name = json.load(f)
    plt.figure()
    model.eval()
    with torch.no_grad():
        for image in os.listdir(data_path):
            image_I = Image.open(os.path.join(data_path, image))
            image_T = transform(image_I)
            image_T = image_T.unsqueeze(0)
            output = model(image_T.to(device))
            output = torch.softmax(output, 1)
            output = output.squeeze()
            output_idx = torch.argmax(output)
            pred = output[output_idx.item()]
            name = cls_name[str(output_idx.item())]
            plt.title("classes: {} score: {}".format(name, pred))
            plt.imshow(image_I)
            plt.show()
            plt.close()

        print("预测结束！")



if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    #预测图片路径
    parser.add_argument('--data-folder', type=str,default=r'./data/sad')
    # 预测的类别个数
    parser.add_argument('--num-classes', type=int, default=9)
    # 权重
    parser.add_argument('--model-pth', type=str, default='./checkpoints/29_resnet34_face.pth')
    # json路径 里面有数据集的类别信息
    parser.add_argument('--json', type=str, default='./classes.json')

    args = parser.parse_args()
    pred()