import os
import shutil


with open('tiny-imagenet-200/val/val_annotations.txt', encoding='UTF-8') as f:
    labels = f.readlines()
    for data in labels:
        data = data.split()
        image = data[0]
        label = data[1]
        if not os.path.isdir('tiny-imagenet-200/val/'+label):
            os.mkdir('tiny-imagenet-200/val/'+label)
            os.mkdir('tiny-imagenet-200/val/'+label+'/images')
        shutil.copyfile('tiny-imagenet-200/val/images/'+image, 'tiny-imagenet-200/val/'+label+'/images/'+image)



