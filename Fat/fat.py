import torch
# -*- coding: utf-8 -*-
import torch.nn as nn
from generation import params
from generation import modules
from generation import funcFAT
from network import SNN
import torchvision
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import statistics
import random

param = params.param
# Initialize a model, and put it on the device specified.
model = SNN.get_model(param.model, param.layers, param.device, param.quant, param.quant_bit)

# The number of batch size.
batch_size = 64
# The number of training epochs.
n_epochs = 4
# If no improvement in 'patience' epochs, early stop.
patience = 5
# For the classification task, we use cross-entropy as the measurement of performance.
criterion = nn.CrossEntropyLoss()

# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)

valid_ratio = 0.2

seed = 777

_exp_name = "sample"

device = param.device

print("-----------------------------------")

#%%# ------------- Utility ------------- #
def train_valid_split(data_set, valid_ratio, seed):
    '''Split provided training data into training set and validation set'''
    valid_set_size = int(valid_ratio * len(data_set)) 
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(seed))
    return train_set, valid_set

# Define a transform
transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((20, 20)),
            torchvision.transforms.Grayscale(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0,), (1,))])

#%%# ------------- Load Data ------------- #
train_data = torchvision.datasets.MNIST(root='data', train=True, download=False, transform=transform)
test_data = torchvision.datasets.MNIST(root='data', train=False, download=False, transform=transform)
train_data, valid_data = train_valid_split(train_data, valid_ratio, seed)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True)

#%%# ------------- Train ------------- #
# Initialize trackers, these are not parameters and should not be changed
stale = 0
best_acc = 0
#* generate fault list using method 1
#* probability default 0.25
#* fault model default Hard to Spike Fault
f_init_list = funcFAT.gen_faultlist(model)

for epoch in range(n_epochs):

    # ---------- Training ----------
    # Make sure the model is in train mode before training.
    model.train()

    # These are used to record information in training.
    train_loss = []
    train_accs = []

    for batch in tqdm(train_loader):

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch
        imgs = imgs.reshape(imgs.shape[0], 20 * 20)

        ##### save weights before back propagation
        original_weights = [p.data.clone() for p in model.parameters()]

        # Forward the data (Make sure data and model are on the same device.)
        logits = model(imgs.to(device))

        # Calculate the cross-entropy loss.
        loss = criterion(logits, labels.to(device))

        # Clear gradients.
        optimizer.zero_grad()

        # Compute gradients.
        loss.backward()

        # Clip gradients for stability.
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

        # Update parameters.
        optimizer.step()

        
        ##### Use gen_faultlist to select faulty neurons and restore their weights to original weights
        for fault in f_init_list:
            # 檢查 fault 的結構
            if len(fault) == 2:
                # 這是非 SWF 的情況
                (layer_id, neuron_id), _ = fault
                # 如果該層的權重是 2D 的，則恢復指定的神經元的權重
                if len(model.layers[layer_id - 1].weight.shape) > 1:
                    # 檢查 neuron_id 是否在範圍內
                    if 0 <= neuron_id < model.layers[layer_id - 1].weight.size(0):
                        model.layers[layer_id - 1].weight.data[neuron_id] = original_weights[layer_id - 1][neuron_id]

            elif len(fault) == 3:
                # 這是 SWF 的情況
                (source_layer_id, source_neuron_id), (target_layer_id, target_neuron_id), w_sa = fault
                # 如果該層的權重是 2D 的，則恢復指定的神經元的權重
                if len(model.layers[source_layer_id].weight.shape) > 1:
                    # 檢查 source_neuron_id 是否在範圍內
                    if 0 <= source_neuron_id < model.layers[source_layer_id].weight.size(0):
                        model.layers[source_layer_id].weight.data[source_neuron_id] = original_weights[source_layer_id][source_neuron_id]

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        train_loss.append(loss.item())
        train_accs.append(acc)

    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)

    # Print the information.
    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

    # ---------- Validation ----------
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    model.eval()

    # These are used to record information in validation.
    valid_loss = []
    valid_accs = []
    # Iterate the validation set by batches.
    for batch in tqdm(valid_loader):

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch
        imgs = imgs.reshape(imgs.shape[0], 20 * 20)

        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            #* input encoding
            input = torch.bernoulli(imgs.to(device))
            logits = model(input)
            
        # We can still compute the loss (but not the gradient).
        loss = criterion(logits, labels.to(device))

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        valid_loss.append(loss.item())
        valid_accs.append(acc.cpu().item())

    # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)

    # Print the information.
    print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

    # update logs
    if valid_acc > best_acc:
        with open(f"./{_exp_name}_log.txt", "a"):
            print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best")
    else:
        with open(f"./{_exp_name}_log.txt", "a"):
            print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

    # save models
    if valid_acc > best_acc:
        print(f"Best model found at epoch {epoch}, saving model")
        # torch.save(model.state_dict(), f"{_exp_name}_best.ckpt") # only save best to prevent output memory exceed error
        best_acc = valid_acc
        stale = 0
    else:
        stale += 1
        if stale > patience:
            print(f"No improvment {patience} consecutive epochs, early stopping")
            break

#* Test the accuracy of trained model
with torch.no_grad():
    model.eval()

    test_accs = []
    inject_accs = []
    total = 0
    correct = 0
    correct_ft = 0

    # Iterate the validation set by batches.
    for batch in tqdm(test_loader):

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch
        imgs = imgs.reshape(imgs.shape[0], 20 * 20)

        input = torch.bernoulli(imgs.to(device))
        logits = torch.reshape(modules.good_inference(model, input), (-1, 10))
        
        # Inject fault
        f_list = funcFAT.gen_faultlist(model)
        bad_infer = torch.reshape(modules.bad_inference(model, input, fault_list=f_list), (-1, 10))
        
        # We can still compute the loss (but not the gradient).
        loss = criterion(logits, labels.to(device))
        b_loss = criterion(bad_infer, labels.to(device))

        # Compute the accuracy for current batch.
        correct += (logits.argmax(dim=-1) == labels.to(device)).sum().item()
        correct_ft += (bad_infer.argmax(dim=-1) == labels.to(device)).sum().item()

        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
        acc_b = (bad_infer.argmax(dim=-1) == labels.to(device)).float().mean()
        total += labels.size(0)
        test_accs.append(acc.cpu().item())
        inject_accs.append(acc_b.cpu().item())

    plt.plot(inject_accs, 'ro', label='faulty')
    plt.plot(test_accs, 'g*', label='fault-free')
    plt.legend()  # add legend to figure
    plt.title('fault-free vs. faulty accuracy')
    plt.savefig('ff_vs_ft.png')  # save figure instead of pop up figure

    fig, ax = plt.subplots()
    bp = ax.boxplot([test_accs, inject_accs], showmeans=True, labels=['fault-free', 'faulty'])
    ax.set_title('accuracy')
    plt.savefig('stats.png')

print(f"Total correctly classified test set images: {correct}/{total}")
print(f"Total correctly classified test set images when faulty: {correct_ft}/{total}")
print(f"Test Set Accuracy (fault-free): {100 * correct / total:.2f}%")
print(f"Test Set Accuracy (  faulty  ): {100 * correct_ft / total:.2f}%")


