import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, recall_score
import numpy as np


class MyModel(nn.Module):
    def __init__(self, num_classes, Trained=False):
        super(MyModel, self).__init__()
        self.resnet = models.resnet18()
        self.resnet.fc = nn.Linear(in_features = self.resnet.fc.in_features, out_features = num_classes)
        if Trained:
            self.resnet.load_state_dict(torch.load("my_model_best_val.pth", map_location=torch.device('cpu')))
            self.resnet.eval()
        
    def forward(self, x):
        return self.resnet(x)
    
    def train(self, loaders, optimizer, criterion, epochs=100, dev=torch.device("cpu"), save_param = False, model_name="resnet18"):
        try:
            self.resnet = self.resnet.to(dev)
            # Initialize history
            history_loss = {"train": [], "val": []}
            history_accuracy = {"train": [], "val": []}
            # Store the best val accuracy
            best_val_accuracy = 0
        

            # Process each epoch
            for epoch in range(epochs):
                # Initialize epoch variables
                sum_loss = {"train": 0, "val": 0}
                sum_accuracy = {"train": 0, "val": 0}


                # Process each split
                for split in ["train", "val"]:
                    if split == "train":
                        self.resnet.train()
                    else:
                        self.resnet.eval()
                    # Process each batch
                    for (input, labels) in loaders[split]:
                        # Move to CUDA
                        input = input.to(dev)
                        labels = labels.to(dev)
                        # Reset gradients
                        optimizer.zero_grad()
                        # Compute output
                        pred = self.resnet(input)
                        loss = criterion(pred, labels)
                        # Update loss
                        sum_loss[split] += loss.item()
                        # Check parameter update
                        if split == "train":
                            # Compute gradients
                            loss.backward()
                            # Optimize
                            optimizer.step()
                        # Compute accuracy
                        _,pred_labels = pred.max(1)
                        batch_accuracy = (pred_labels == labels).sum().item()/input.size(0)
                        # Update accuracy
                        sum_accuracy[split] += batch_accuracy
                # Compute epoch loss/accuracy
                epoch_loss = {split: sum_loss[split]/len(loaders[split]) for split in ["train", "val"]}
                epoch_accuracy = {split: sum_accuracy[split]/len(loaders[split]) for split in ["train", "val"]}

                # Store params at the best validation accuracy
                if save_param and epoch_accuracy["val"] > best_val_accuracy:
                    torch.save(self.resnet.state_dict(), f"{model_name}_best_val.pth")
                    best_val_accuracy = epoch_accuracy["val"]

                # Update history
                for split in ["train", "val"]:
                    history_loss[split].append(epoch_loss[split])
                    history_accuracy[split].append(epoch_accuracy[split])
                # Print info
                print(f"Epoch {epoch+1}:",
                    f"TrL={epoch_loss['train']:.4f},",
                    f"TrA={epoch_accuracy['train']:.4f},",
                    f"VL={epoch_loss['val']:.4f},",
                    f"VA={epoch_accuracy['val']:.4f},",)
                
    
        

        except KeyboardInterrupt:
            print("Interrupted")
        finally:     
            # Plot loss
            plt.title("Loss")
            for split in ["train", "val"]:
                plt.plot(history_loss[split], label=split)
            plt.legend()
            plt.show()
            # Plot accuracy
            plt.title("Accuracy")
            for split in ["train", "val"]:
                plt.plot(history_accuracy[split], label=split)
            plt.legend()
            plt.show()



    # Test
    def test(self, loaders, criterion, dev=torch.device("cpu")):
        self.resnet = self.resnet.to(dev)
        true_labels = []
        predicted_labels = []
        sum_loss = 0
        sum_accuracy = 0


        with torch.no_grad():
            # Process each batch
            for (input, labels) in loaders["test"]:
                # Move to CUDA
                input = input.to(dev)
                labels = labels.to(dev)
                # Compute output
                pred = self.resnet(input)
                loss = criterion(pred, labels)
                # Update loss
                sum_loss += loss.item()

                # Compute accuracy
                _,pred_labels = pred.max(1)
                batch_accuracy = (pred_labels == labels).sum().item()/input.size(0)
                # Update accuracy
                sum_accuracy += batch_accuracy
                # Collect true labels and predicted labels for confusion matrix
                true_labels.extend(labels.cpu().numpy())
                predicted_labels.extend(pred_labels.cpu().numpy())

        # Compute epoch loss/accuracy
        test_loss = sum_loss/len(loaders["test"])
        test_accuracy = sum_accuracy/len(loaders["test"]) 

        confusion = confusion_matrix(true_labels, predicted_labels)
        num_classes = len(np.unique(true_labels))

        # Normalize the confusion matrix
        confusion_norm = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]
        
        # Calculate F1 score and recall
        f1 = f1_score(true_labels, predicted_labels, average='macro')
        recall = recall_score(true_labels, predicted_labels, average='macro')

        return test_loss, test_accuracy, confusion_norm, num_classes, f1, recall
