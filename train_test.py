import torch
import numpy as np
import sys


def calculate_accuracy(outputs, ground_truth):
    softmaxed_output = torch.nn.functional.softmax(outputs, dim=1)
    predictions = torch.argmax(softmaxed_output, dim=1)
    num_correct = torch.sum(torch.eq(predictions, ground_truth)).item()
    return num_correct, ground_truth.size(0)

def train(model: torch.nn.Module, num_epochs: int, optimizer, loss_function, trainloader, validloader, device, path_to_model='etinynet', scheduler=None) -> torch.nn.Module:
    print('Training Started')
    sys.stdout.flush()
    for epoch in range(1, num_epochs+1):
        train_loss = []
        valid_loss = []
        num_correct_train = 0
        num_correct_valid = 0
        num_examples_train = 0
        num_examples_valid = 0
        model.train()
        for batch in trainloader:
            optimizer.zero_grad()
            images = batch[0].to(device)
            labels = batch[1].to(device)
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            num_corr, num_ex = calculate_accuracy(outputs, labels)
            num_examples_train += num_ex
            num_correct_train += num_corr
        
        model.eval()
        with torch.no_grad():
            for batch in validloader:
                images = batch[0].to(device)
                labels = batch[1].to(device)
                outputs = model(images)
                loss = loss_function(outputs, labels)
                valid_loss.append(loss.item())
                num_corr, num_ex = calculate_accuracy(outputs, labels)
                num_examples_valid += num_ex
                num_correct_valid += num_corr
        

        print(f'Epoch: {epoch}, Training Loss: {np.mean(train_loss)}, Validation Loss: {np.mean(valid_loss)}, Training Accuracy: {num_correct_train/num_examples_train}, Validation Accuracy: {num_correct_valid/num_examples_valid}')
        
        if scheduler:
            scheduler.step(np.mean(valid_loss))
            print(f'Last Scheduler Learning Rate: {scheduler.get_last_lr()}')
    
    torch.save(model, path_to_model+'.pth')
    return model

@torch.no_grad
def test(model: torch.nn.Module, testloader, loss_function, device)-> torch.nn.Module:
    test_loss = []
    num_examples = 0
    num_correct = 0
    model.eval()
    for batch in testloader:
        images = batch[0].to(device)
        labels = batch[1].to(device)
        output = model(images)
        loss = loss_function(output, labels)
        test_loss.append(loss.item())
        num_corr, num_ex = calculate_accuracy(output, labels)
        num_examples += num_ex
        num_correct += num_corr
    
        print(f'Test Loss: {np.mean(test_loss)}, Test Accuracy: {num_correct/num_examples}')
    return model