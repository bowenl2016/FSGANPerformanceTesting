import torch
from torchvision import transforms
from model_resnet import resnet34
from model import AlexNet
from torchvision import datasets
import os
import json
import torch.nn as nn
from tqdm import tqdm
from plot_loss_acc import plot_loss, plot_acc


#保存模型权重
def save_model(model,args,epoch):
    save_dir = args.checkpoints + "/{}_{}_{}.pth".format(epoch,args.model_name,args.data_name)
    print("save model ckpt in {}".format(save_dir))
    if not os.path.exists(args.checkpoints):
        os.mkdir(args.checkpoints)
    torch.save(model.state_dict(), save_dir)


def main():

    #设备选用gpu 或者cpu训练
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device {}".format(device))
    #数据预处理
    data_transform = {"train":transforms.Compose([transforms.Resize((224,224)),  # 将图片全部resize到224
                                         transforms.RandomHorizontalFlip(),   #随机水平翻转
                                         transforms.ToTensor(),            # 转化为tensor
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]), #标准化 归一化
                      "val": transforms.Compose([transforms.Resize((224,224)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    data_path = args.data_folder
    assert data_path,"{} does not exist".format(data_path)
    #读取数据集 分为训练和验证
    train_dataset = datasets.ImageFolder(root=os.path.join(data_path, "train"),transform=data_transform['train'])
    val_dataset = datasets.ImageFolder(root=os.path.join(data_path, "val"), transform=data_transform['val'])
    train_num = len(train_dataset)
    val_num = len(val_dataset)
    print("using {} images to train, using {} iamges to validation!".format(train_num, val_num))
    #得到类别信息
    cls = train_dataset.class_to_idx
    cls = dict((key, val) for (val,key) in cls.items())
    json_str = json.dumps(cls,ensure_ascii=False,indent=4)
    #将类别信息 写入到json中
    with open("./classes.json","w", encoding='utf-8') as f:
        f.write(json_str)
    # 打包成一个一个的batch
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    #模型定义
    # model = resnet34()
    #
    # #迁移学习 需要设置resume为true 并且pretrain 填写预训练权重的路径
    # model_pretrain = args.pretrain
    # if args.resume:
    #     model.load_state_dict(torch.load(model_pretrain))
    # #print(model)
    # # 把最后一层全连接的输出给换成自己的输出类别个数
    # in_channel = model.fc.in_features
    # model.fc = nn.Linear(in_channel, args.num_classes)
    model = AlexNet(num_classes=args.num_classes)
    model.to(device)
    # 损失函数默认求的是平均损失 所以轮次损失除以总批次而不是总图片个数
    # 多分类损失函数
    loss_function = nn.CrossEntropyLoss()
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
    best_acc = 0
    loss_list = [[] for i in range(2)]
    acc_list = [[] for i in range(2)]
    #训练开始 epoch 轮数
    step1 = 0
    step2 = 0
    for epoch in range(args.epochs):
        train_bar = tqdm(train_loader)
        print(len(train_bar))
        model.train()
        train_epoch_loss = 0
        train_epoch_acc = 0
        for i , (images,labels) in enumerate(train_bar):

            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            #输入模型
            output = model(images)
            #计算损失
            loss = loss_function(output, labels)
            #反向传播
            loss.backward()
            optimizer.step()

            output_idx = torch.argmax(output, dim=1)
            acc = torch.eq(output_idx,labels).sum().item()
            train_epoch_acc += acc
            train_epoch_loss += loss.item()
            train_bar.desc = "epoch: [{}/{}] train_loss: {:.3f}".format(epoch,args.epochs,loss)
            step1 = epoch * len(train_bar) + i


        #验证阶段
        model.eval()
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            val_epoch_loss = 0
            val_epoch_acc = 0
            for i,(images, labels) in enumerate(val_bar):
                images,labels = images.to(device), labels.to(device)
                output = model(images)
                loss = loss_function(output, labels)
                val_epoch_loss += loss.item()
                output_idx = torch.argmax(output,dim=1)
                acc = torch.eq(output_idx,labels).sum().item()
                val_epoch_acc += acc

                val_bar.desc = "epoch: [{}/{}]  val_loss: {:.3f}".format(epoch, args.epochs, loss)
                step2 = epoch * len(val_bar) + i


        train_epoch_loss = train_epoch_loss / len(train_loader)
        train_epoch_acc = train_epoch_acc / train_num
        val_epoch_loss = val_epoch_loss / len(val_loader)
        val_epoch_acc = val_epoch_acc / val_num
        loss_list[0].append(train_epoch_loss)
        loss_list[1].append(val_epoch_loss)
        acc_list[0].append(train_epoch_acc)
        acc_list[1].append(val_epoch_acc)

        if epoch >0 and epoch % 5==0:
            plot_loss(loss_list, "a3", args)
            plot_acc(acc_list, "b3", args)

        print("train_loss_epoch: {:.3f} train_acc_epoch: {:.3f} val_loss_epoch: {:.3f} val_acc_epoch: {:.3f}".format(train_epoch_loss,train_epoch_acc,val_epoch_loss,val_epoch_acc))

        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            save_model(model,args,epoch)

    print("train done!!")

if __name__ =="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    #数据集路径
    parser.add_argument("--data-folder",type=str,default=r"./data")
    # 数据集名字
    parser.add_argument('--data-name', type=str, default="face3")
    # batch大小设置
    parser.add_argument('--batch-size',type=int,default=8)
    # 类别个数
    parser.add_argument('--num-classes', type=int, default=9)
    # 是否迁移学习
    parser.add_argument('--resume', type=bool, default=False)
    # 迁移学习的 权重路径
    parser.add_argument('--pretrain', type=str, default='./resnet34-pre.pth')
    # 学习率设置
    parser.add_argument('--lr', type=float, default=1e-4)
    # 训练轮数设置
    parser.add_argument('--epochs', type=int, default=31)
    # 保存的训练权重路径
    parser.add_argument('--checkpoints', type=str, default='./checkpoints')
    # 保存的训练曲线路径
    parser.add_argument('--fig-dir', type=str, default='./fig')
    #模型名字
    parser.add_argument('--model-name', type=str, default='alexnet')


    args = parser.parse_args()
    if not os.path.exists(args.fig_dir):
        os.mkdir(args.fig_dir)
    main()