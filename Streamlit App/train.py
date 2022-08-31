from tqdm import tqdm
import torch

def train(num_epoch, model,train_loader, loss_in, optimizer):
    model
    for epoch in range(0, num_epoch):
        model.train()
        loop = tqdm(enumerate(train_loader), total=len(train_loader)) # create a progress bar
        for batch_idx, (ancre, lab_ancre) in loop:
            out = model(ancre)
            loss = loss_in(out, lab_ancre)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            _, preds_ancre = torch.max(out, 1)

            loop.set_description(f"Epoch {epoch+1}/{num_epoch} process: {int((batch_idx / len(train_loader)) * 100)}")
            loop.set_postfix(loss=loss.data.item())
