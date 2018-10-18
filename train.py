import argparse
import os.path

import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict

def do_deep_learning(model, trainloader, testloader, validloader, epochs, print_every, criterion, optimizer, device='cpu'):
    steps = 0
    print(model)

    model.to(device)

    for e in range(epochs):
        model.train()
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            #print(ii)
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            #print(outputs.size())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if steps % print_every == 0:
                model.eval()
                accuracy = 0
                validation_loss  = 0
                validation_loss,accuracy = validation(model,validloader,criterion,device)
                print("Epoch: {}/{}... ".format(e+1, epochs),
                     "| Training Loss: {:.4f}".format(running_loss / print_every),
                     "| Validation Loss: {:.3f}.. ".format(validation_loss  / len(validloader)),
                     "| Validation Accuracy: {:.3f}%".format(accuracy / len(validloader) * 100))
                running_loss = 0
                model.train()
    return model

def validation(model, testloader, criterion, device):
    test_loss = 0
    accuracy = 0
    for images, labels in testloader:
        inputs, labels = images.to(device), labels.to(device)
        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy

def getDataLoader(train_dir,test_dir,valid_dir):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)

    return [trainloader,testloader,validloader],[train_data,test_data,valid_data]

def setClassifier(model,layers):
    #freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    in_features_num = model.classifier[0].in_features
    layerDict = OrderedDict()
    i=0
    layers = [in_features_num] + layers
    for i in range(len(layers)-1):
        layerDict.update({'fc%d' %i:nn.Linear(layers[i],layers[i+1])})
        layerDict.update({'relu%d' %i: nn.ReLU()}),
        layerDict.update({'dropout%d' %i: nn.Dropout(p = 0.5)})
    layerDict.update({'fcFinal':nn.Linear(layers[len(layers)-1],102)})
    layerDict.update({'output':nn.LogSoftmax(dim = 1)})

    classifier = nn.Sequential(layerDict)

    model.classifier = classifier

    return model


def validateModel(model,testloader,device):
    model.to(device)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    print(total)
    print(correct)

def load_checkpoint(filepath,arch):
    checkpoint = torch.load(filepath)
    model = getattr(models, arch)(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model

def saveModel(model,train_data,save_file,arch,epochs,learn_rate):
    model.class_to_idx = train_data.class_to_idx

    checkpoint = {
       'epochs': 10,
       'learn_rate': 0.001,
       'optimizer_type': 'Adam',
       'output_size': 102,
       'state_dict': model.state_dict(),
       'class_to_idx': train_data.class_to_idx,
       'classifier': model.classifier,
       'criterion_type': 'NLLLoss',
       'pretrain_model':arch
    }
    torch.save(checkpoint, save_file)
    
def main():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('data_dir', help = 'Data directory')
    parser.add_argument('--save_dir',type = str,help='Directory to save checkpoint')
    parser.add_argument('--arch',type = str,default='vgg16',help='choose model')
    parser.add_argument('--learning_rate',type = float,default=0.001,help='model learning_rate')
    parser.add_argument('--hidden_units',default=[512],nargs='+',type=int,help='Number of hidden layers')
    parser.add_argument('--epochs',type = int,default=5,help='Numbers of epochs train')
    parser.add_argument('--gpu',action='store_true', default=False,help='Use gpu for training')
    args = parser.parse_args()

    # data_dir = '/home/workspace/aipnd-project/flowers'
    data_dir = args.data_dir
    train_dir = args.data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    save_file = args.save_dir+'/checkpoint.pth' if args.save_dir else 'checkpoint.pth'
    print(save_file)
    
    dataloader,dataTrain = getDataLoader(train_dir,test_dir,valid_dir)
    
    if(os.path.isfile(save_file)):
        model = load_checkpoint(save_file,args.arch)
        print(model)
    else:
        model = getattr(models, args.arch)(pretrained=True)
        model = setClassifier(model,args.hidden_units)
    
    device = 'cuda:0' if args.gpu else 'cpu'
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    
    model = do_deep_learning(model,dataloader[0],dataloader[1],dataloader[2],args.epochs,20, criterion, optimizer,device)
    validateModel(model,dataloader[1],device)
    saveModel(model,dataTrain[0],save_file,args.arch,args.epochs,args.learning_rate)

if __name__ == '__main__':
    main()
 