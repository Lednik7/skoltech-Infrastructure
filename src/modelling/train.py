import torch
from tqdm.notebook import tqdm


def train(model, train_loader, optimizer, scheduler, criterion, metric, device):
    model.train()
    total_score = 0
    total_loss = 0
    for inputs, targets in tqdm(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss
        total_score += metric(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
    return {"loss": total_loss / len(train_loader),
            "score": total_score / len(train_loader)}


def validate(model, val_loader, criterion, metric, device):
    model.eval()
    total_loss = 0
    total_score = 0
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs, targets)
            total_score += metric(outputs, targets)
    return {"loss": total_loss / len(val_loader),
            "score": total_score / len(val_loader)}


def fit_loop(model, train_loader, val_loader, optimizer, scheduler, criterion, metric,
             epochs,
             device):
    for epoch in range(1, epochs + 1):
        train_stats = train(model, train_loader, optimizer, scheduler, criterion,
                            metric, device)
        print(f"Epoch {epoch}")
        print(f"Train loss: {train_stats['loss']}, score: {train_stats['score']}")
        val_stats = validate(model, val_loader, criterion, metric, device)
        print(f"Val loss: {val_stats['loss']}, score: {val_stats['score']}")
