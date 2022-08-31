from tqdm import tqdm
import torch
from test import test

def train(num_epoch, model, train_loader, test_loader, loss_in, optimizer):
    accuracy_list = []
    for epoch in range(0, num_epoch):
        model.train()
        loop = tqdm(enumerate(train_loader), total=len(train_loader)) # create a progress bar
        for batch_idx, (ancre, lab_ancre) in loop:
            out = model(ancre)
            loss = loss_in(out, torch.tensor(lab_ancre, dtype=torch.int64))
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            _, preds_ancre = torch.max(out, 1)

            loop.set_description(f"Epoch {epoch+1}/{num_epoch} process: {int((batch_idx / len(train_loader)) * 100)}")
            loop.set_postfix(loss=loss.data.item())
        
        loss, accuracy = test(model, test_loader, loss_in)
        #print("\nAccuracy : ", accuracy, " Loss : ", loss, "\n")
        accuracy_list.append(accuracy)
    return accuracy_list

def train_without_test(num_epoch, model, train_loader, test_loader, loss_in, optimizer):
    accuracy_list = []
    for epoch in range(0, num_epoch):
        model.train()
        loop = tqdm(enumerate(train_loader), total=len(train_loader)) # create a progress bar
        for batch_idx, (ancre, lab_ancre) in loop:
            out = model(ancre)
            loss = loss_in(out, torch.tensor(lab_ancre, dtype=torch.int64))
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            _, preds_ancre = torch.max(out, 1)

            loop.set_description(f"Epoch {epoch+1}/{num_epoch} process: {int((batch_idx / len(train_loader)) * 100)}")
            loop.set_postfix(loss=loss.data.item())
        
        #loss, accuracy = test(model, test_loader, loss_in)
        #print("\nAccuracy : ", accuracy, " Loss : ", loss, "\n")
        #accuracy_list.append(accuracy)
    return accuracy_list

        
