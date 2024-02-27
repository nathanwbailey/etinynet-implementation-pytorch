import torch
import torchvision
from EtinyNet import EtinyNet
from train_test import train
from train_test import test
from torchvision.datasets import ImageFolder

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

fixed_generator = torch.Generator().manual_seed(42)

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224,224)),
    torchvision.transforms.ToTensor()
])


train_dataset = ImageFolder('tiny-imagenet-200/train', transforms)


trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True, num_workers=4)

mean = torch.Tensor([0.4805, 0.4483, 0.3978]).to('cpu')
std = torch.Tensor([0.2647, 0.2569, 0.2701]).to('cpu')

# mean = torch.zeros(3).to(device)
# std = torch.zeros(3).to(device)

# for idx, batch in enumerate(trainloader):
#     image = batch[0].to(device)
#     image_mean = torch.mean(image, dim=(0,2,3))
#     image_std = torch.std(image, dim=(0,2,3))
#     mean = torch.add(mean, image_mean)
#     std = torch.add(std, image_std)

# mean = (mean/len(trainloader)).to('cpu')
# std = (std/len(trainloader)).to('cpu')

print(mean)
print(std)

train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224,224)),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean, std)
])

test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224,224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean, std)
])

train_dataset = ImageFolder('tiny-imagenet-200/train', train_transforms)
valid_dataset = ImageFolder('tiny-imagenet-200/val', test_transforms)

valid_set, test_set = torch.utils.data.random_split(valid_dataset, [0.7, 0.3], generator=fixed_generator)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, pin_memory=True, num_workers=4)
validloader = torch.utils.data.DataLoader(valid_set, batch_size=128, shuffle=False, pin_memory=True, num_workers=4)
testloader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, pin_memory=True, num_workers=4)

print(len(trainloader))
print(len(validloader))
print(len(testloader))

etinynet_block_info = [
    {
        "block_type": "lb",
        "layer_values": [{"in_channels": 32, "out_channels": 32} for _ in range(4)]
    },
    {
        "block_type": "lb",
        "layer_values": [{"in_channels": 32, "out_channels": 128}] + [{"in_channels": 128, "out_channels": 128} for _ in range(3)]
    },
    {
        "block_type": "dlb",
        "layer_values": [{"in_channels": 128, "out_channels": 192}] + [{"in_channels": 192, "out_channels": 192} for _ in range(2)]
    },
    {
        "block_type": "dlb",
        "layer_values": [{"in_channels": 192, "out_channels": 256}, {"in_channels": 256, "out_channels": 256}, {"in_channels": 256, "out_channels": 512}]
    }
]

network = EtinyNet(block_info=etinynet_block_info).to(device)
torch.autograd.set_detect_anomaly(True)
loss = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(filter(lambda param: param.requires_grad, network.parameters()), lr = 0.1, momentum=0.9, weight_decay=1e-4)

# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.1, mode='min', patience=3, min_lr=1e-7, threshold_mode='abs', threshold=1e-4)
num_epochs = 1000

network = train(model=network, num_epochs=num_epochs, optimizer=optimizer, loss_function=loss, trainloader=trainloader, validloader=validloader, device=device, scheduler=scheduler)

test_loss = torch.nn.CrossEntropyLoss()

network = test(model=network, testloader=testloader, loss_function=test_loss, device=device)

