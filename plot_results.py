import matplotlib.pyplot as plt


def plot(file, name):

    training_stats = open(file, 'r')
    train_loss = []
    valid_loss = []
    train_accuracy = []
    valid_accuracy = []
    train_accuracy_5 = []
    valid_accuracy_5 = []
    for i in training_stats:
        if 'Validation Accuracy' in i:
            try:
                y = i.split()
                train_loss.append(float(y[4].strip(',')))
                valid_loss.append(float(y[7].strip(',')))
                train_accuracy.append(float(y[10].strip(',')))
                valid_accuracy.append(float(y[13].strip(',')))
                train_accuracy_5.append(float(y[18].strip(',')))
                valid_accuracy_5.append(float(y[23].strip(',')))
            except IndexError:
                continue
    epochs = [i for i in range(1, len(train_loss)+1)]
    plt.clf()
    plt.plot(epochs, valid_loss, '-c', label='Validation Loss')
    plt.plot(epochs, train_loss, '-m', label='Train Loss')

    plt.title('Loss vs Epoch')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('Loss '+name+'.png')
    plt.clf()

    plt.plot(epochs, valid_accuracy, '-c', label='Validation Accuracy')
    plt.plot(epochs, train_accuracy, '-m', label='Train Accuracy')
    plt.title('Top-1 Accuracy vs Epoch')
    plt.ylabel('Top-1 Accuracy')

    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('Top-1 Accuracy '+name+'.png')
    plt.clf()


    plt.plot(epochs, valid_accuracy_5, '-c', label='Validation Accuracy')
    plt.plot(epochs, train_accuracy_5, '-m', label='Train Accuracy')
    plt.title('Top-5 Accuracy vs Epoch')
    plt.ylabel('Top-5 Accuracy')

    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('Top-5 Accuracy '+name+'.png')


plot('overfitting_run_dropout_0.1.out', 'dropout_0.1')
plot('overfitting_run_dropout_0.2.out', 'dropout_0.2')
plot('overfitting_run_dropout_0.3.out', 'dropout_0.3')
plot('overfitting_run_dropout_0.4.out', 'dropout_0.4')
plot('overfitting_run_dropout_0.5.out', 'dropout_0.5')