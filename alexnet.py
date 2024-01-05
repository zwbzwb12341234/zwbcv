import torch.nn as nn
import torch 
import torch
import torch.nn as nn 
import os  
from torchvision import transforms, datasets 
from torch.utils.data import DataLoader
import time
import torch.optim as optim   

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000 ):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[96, 55, 55]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[96, 27, 27]

            nn.Conv2d(96, 256, kernel_size=5, padding=2),           # output[256, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[256, 13, 13]

            nn.Conv2d(256, 384, kernel_size=3, padding=1),          # output[384, 13, 13]
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, padding=1),          # output[384, 13, 13]
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),          # output[256, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[256, 6, 6]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            
            nn.Linear(4096, num_classes),
        )
  
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x
 


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('using {} device.'.format(device))
    
    data_transform = {
        "train": transforms.Compose(
            [transforms.RandomResizedCrop(224),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(), #把shape=(H x W x C) 的像素值为 [0, 255] 的 PIL.Image 和 numpy.ndarray转换成shape=(C,H,WW)的像素值范围为[0.0, 1.0]的 torch.FloatTensor
             transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
             ]
        ),
        "test":transforms.Compose(
            [transforms.Resize((224,224)),
             transforms.ToTensor(),
             transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
             ]
        )
    }
    
    data_path = os.path.abspath(os.path.join("D:\\VScode\\Datasets\\flower")) 
    train_dataset = datasets.ImageFolder(root=os.path.join(data_path, "train"), transform=data_transform["train"]) 
    vaild_dataset = datasets.ImageFolder(root=os.path.join(data_path, "val"), transform=data_transform["test"])
    train_num = len(train_dataset)
    test_num = len(vaild_dataset)
 
    batch_size = 16
    n_works = min([os.cpu_count(), batch_size if batch_size >1 else 0, 9])
    print('Using {} dataloader workers every process'.format(n_works))
    
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=n_works, drop_last=True)
    valid_loader = DataLoader(vaild_dataset, batch_size=4, shuffle=False, num_workers=n_works, drop_last=True)
    
    print('using {} images for traing and using {} images for testing'.format(train_num, test_num))
    
    net = AlexNet(num_classes=5)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0002)
    
    epochs = 40  
    save_path = os.path.join(os.getcwd(), 'checkpoints\\alex')
    if os.path.isdir(save_path):
        print("checkpoints save in " + save_path)
    else:
        os.makedirs(save_path)
        print("new a dir to save checkpoints: " + save_path)
        
    best_acc = 0.0
    train_steps = len(train_loader)  
 
    # training  
    for epoch in range(epochs):
        time_start = time.time()
        net.train()
        running_loss = 0.0  
        for step, data in enumerate(train_loader):
            images, labels = data 
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            print("Epoch" +str(epoch)+ ": processing:" + str(step) + "/" + str(train_steps))
            

        # validate
        time_end = time.time()
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad(): 
            for val_data in valid_loader:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / test_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f time_one_epoch: %.3f '  %
              (epoch + 1, running_loss / train_steps, val_accurate, (-time_start+time_end)))
 
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), os.path.join(save_path, 'alex_flower.pth'))

    print('Finished Training')


if __name__ == '__main__':
    main()
