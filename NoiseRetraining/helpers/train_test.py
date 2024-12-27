import torch

def train(model, device, trainloader, criterion, optimizer, slurm_batch_mode):
    model.train()
    train_correct = 0
    train_loss = 0.0

    for i, (x, y) in enumerate(trainloader):
        optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
        _, predicted_train = outputs.max(1)
        train_correct += predicted_train.eq(y).sum().item()
        
        if not slurm_batch_mode:
            print(f"\rBatch: {i+1}/{len(trainloader)}", end="")

    return train_correct, train_loss


def test(model, device, testloader, criterion):
    model.eval()
    test_correct = 0
    test_loss = 0.0

    with torch.no_grad():
        for x_val, y_val in testloader:
            x_val = x_val.to(device)
            y_val = y_val.to(device)
            val_outputs = model(x_val)
            test_loss += criterion(val_outputs, y_val).item()
            
            _, predicted_test = val_outputs.max(1)
            test_correct += predicted_test.eq(y_val).sum().item()

    return test_correct, test_loss