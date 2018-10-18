import argparse
import json
import torch
import numpy as np
from torchvision import  models
from PIL import Image

def process_image(image): 
    size=256,256
    new_size=224
    im=Image.open(image)
    im.thumbnail(size)
    im = im.crop((0,0,new_size,new_size))
    image = np.array(im)
    np_image = image/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/std
    np_image = np_image.transpose((2,0,1))
    
    return np_image

def predict(image_path, model, device, topk=5):
    model.to(device)
    #Convert Image File to NP Tensor
    model.eval()
    np_image = process_image(image_path)
    img_as_tensor = torch.from_numpy(np.array(np_image)).float()
    input = torch.FloatTensor(img_as_tensor).to(device)

    input.unsqueeze_(0)
    
    output= model.forward(input)
    print(output.size())

    probabilities, classes = torch.exp(output).topk(topk)
    return probabilities,classes

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = getattr(models,checkpoint['pretrain_model'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model


def main():
    parser = argparse.ArgumentParser(description='predict')
    parser.add_argument('input', help = 'Path to imagen')
    parser.add_argument('checkpoint',help='Save model to load, path to the model')
    parser.add_argument('--top_k',type = int,default=5,help='K most likely classes')
    parser.add_argument('--category_names',type = str,default='cat_to_name.json',help='mapping categories')
    parser.add_argument('--gpu',action='store_true', default=False,help='Use gpu for inference')
    args = parser.parse_args()
    
    device = 'cuda:0' if args.gpu else 'cpu'
    
    model = load_checkpoint(args.checkpoint)
    #print(model)

    probs, classes = predict(args.input, model, device, args.top_k)

    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    if args.gpu:
        classes = classes.detach().cpu().numpy()[0]
        probs = probs.detach().cpu().numpy()[0]
    else:
        classes = classes.detach().numpy()[0]
        probs = probs.detach().numpy()[0]

    TopKClasses = [cat_to_name[str(clas+1)] for clas in classes]

    print(probs)
    print(TopKClasses)

if __name__ == '__main__':
    main()