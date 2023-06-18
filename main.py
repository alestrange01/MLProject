from ChestXrayDataset import ChestXrayDataset
from MyModel import MyModel
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    print("Caricamento dataset...")
    # Compose transformations
    dataset = ChestXrayDataset(img_dir='/Users/ale_strange/Desktop/dvntn9yhd2-1/aunione')
    print("Dataset caricato")

    num_train = dataset.get_trainset().__len__()
    num_test = dataset.get_testset().__len__()
    num_val = dataset.get_valset().__len__()

    print('Numero di immagini nel train set: ', num_train)
    print('Numero di immagini nel validation set: ', num_val)
    print('Numero di immagini nel test set: ', num_test)

    # Creare i DataLoader per ogni set
    batch_size = 32
    #setup device to use the GPU
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = dataset.get_loaders(batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2) 

    print("Vuoi allenare il modello o testarlo?")
    print("1. Allenare il modello")
    print("2. Testare il modello")
    #scansione dell'input
    scelta = input()
    while scelta != "1" and scelta != "2":
        print("Input non valido")
        scelta = input()


    if scelta == "1":
        resnet18 = MyModel(num_classes=3, Trained=False)
        # Define an optimizer
        optimizer = optim.SGD(resnet18.fc.parameters(), lr = 0.005, weight_decay = 0.008)
        # Define a loss
        criterion = nn.CrossEntropyLoss()
        # Train the model
        resnet18.train_model(loaders, optimizer, criterion, num_epochs=200, save_param = True, dev=dev)
    elif scelta == "2":
        resnet18 = MyModel(num_classes=3, Trained=True)
        # Test the model
        print("Modello caricato")
        test_loss, test_accuracy, confusion_matrix, num_classes = resnet18.test(loaders, criterion=nn.CrossEntropyLoss(), dev=dev)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        print("Chiudere la matrice di confusione per continuare")
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_matrix, annot=True, fmt=".2f", cmap="Blues", xticklabels=num_classes, yticklabels=num_classes)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()
        print("Test completato")


if __name__ == "__main__":
    main()