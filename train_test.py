import torch
import numpy as np
import sys
from early.pytorchtools import EarlyStopping


def calculate_accuracy(outputs, ground_truth):
    softmaxed_output = torch.nn.functional.softmax(outputs, dim=1)
    predictions = torch.argmax(softmaxed_output, dim=1)
    num_correct = torch.sum(torch.eq(predictions, ground_truth)).item()
    return num_correct, ground_truth.size(0)

def calculate_accuracy_top_5(outputs, ground_truth):
    num_correct = 0
    softmaxed_output = torch.nn.functional.softmax(outputs, dim=1)
    predictions = torch.argsort(softmaxed_output, dim=1, descending=True)
    for idx, x in enumerate(ground_truth):
        if torch.isin(x, predictions[idx, :4]):
            num_correct += 1
    return num_correct, ground_truth.size(0)

def train(model: torch.nn.Module, num_epochs: int, optimizer, loss_function, trainloader, validloader, device, path_to_model='etinynet', scheduler=None, patience=10) -> torch.nn.Module:
    print('Training Started')
    early_stop = EarlyStopping(patience=patience, verbose=True, path=path_to_model+'.pt')
    for epoch in range(1, num_epochs+1):
        train_loss = []
        valid_loss = []
        num_correct_train = 0
        num_correct_valid = 0
        num_examples_train = 0
        num_examples_valid = 0
        num_correct_train_5 = 0
        num_correct_valid_5 = 0
        num_examples_train_5 = 0
        num_examples_valid_5 = 0
        model.train()
        for _, batch in enumerate(trainloader):
            optimizer.zero_grad()
            images = batch[0].to(device)
            labels = batch[1].to(device)
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            num_corr, num_ex = calculate_accuracy(outputs, labels)
            num_corr_5, num_ex_5 = calculate_accuracy_top_5(outputs, labels)
            num_examples_train += num_ex
            num_correct_train += num_corr
            num_examples_train_5 += num_ex_5
            num_correct_train_5 += num_corr_5
        
        model.eval()
        with torch.no_grad():
            for batch in validloader:
                images = batch[0].to(device)
                labels = batch[1].to(device)
                outputs = model(images)
                loss = loss_function(outputs, labels)
                valid_loss.append(loss.item())
                num_corr, num_ex = calculate_accuracy(outputs, labels)
                num_corr_5, num_ex_5 = calculate_accuracy_top_5(outputs, labels)
                num_examples_valid += num_ex
                num_correct_valid += num_corr
                num_examples_valid_5 += num_ex_5
                num_correct_valid_5 += num_corr_5

        print(f'Epoch: {epoch}, Training Loss: {np.mean(train_loss)}, Validation Loss: {np.mean(valid_loss)}, Training Accuracy: {num_correct_train/num_examples_train}, Validation Accuracy: {num_correct_valid/num_examples_valid}, Top 5 Training Accuracy: {num_correct_train_5/num_examples_train_5}, Top 5 Validation Accuracy: {num_correct_valid_5/num_examples_valid_5}')
        
        if scheduler:
            scheduler.step(np.mean(valid_loss))
            print(f'Last Scheduler Learning Rate: {scheduler.get_last_lr()}')
    
        if early_stop.counter == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'validation_loss': np.mean(valid_loss),
                'train_loss': np.mean(train_loss)
                }, path_to_model+'_complete_collection.tar')
            torch.save(model, path_to_model+'_full_model.pth')
        
        early_stop_loss = np.mean(valid_loss)
        early_stop(early_stop_loss, model)
        
        if early_stop.early_stop:
            print(f'Early Stopping at Epoch: {epoch}')
            break
            
    model.load_state_dict(torch.load(path_to_model+'.pt'))
    model.to(device)
    return model

@torch.no_grad
def test(model: torch.nn.Module, testloader, loss_function, device)-> torch.nn.Module:
    test_loss = []
    num_examples = 0
    num_correct = 0
    num_examples_5 = 0
    num_correct_5 = 0
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
        num_corr_5, num_ex_5 = calculate_accuracy_top_5(output, labels)
        num_examples_5 += num_ex_5
        num_correct_5 += num_corr_5
    
    print(f'Test Loss: {np.mean(test_loss)}, Test Accuracy: {num_correct/num_examples}, Test Accuracy Top 5: {num_correct_5/num_examples_5}')
    return model