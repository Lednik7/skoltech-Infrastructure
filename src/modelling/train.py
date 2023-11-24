import os
import time

import torch
import tqdm
from tqdm.notebook import tqdm


# helping function to normal visualisation in Colaboratory
def foo_():
    time.sleep(0.3)


def train_epoch(model, train_dl, criterion, metric, optimizer, scheduler, device):
    model.train()
    loss_sum = 0
    score_sum = 0
    for X, y in tqdm(train_dl):
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss = loss.item()
        score = metric(output > 0.5, y).mean().item()
        loss_sum += loss
        score_sum += score
    return loss_sum / len(train_dl), score_sum / len(train_dl)


def eval_epoch(model, val_dl, criterion, metric, device):
    model.eval()
    loss_sum = 0
    score_sum = 0
    for X, y in tqdm(val_dl):
        X = X.to(device)
        y = y.to(device)

        with torch.no_grad():
            output = model(X)
            loss = criterion(output, y).item()
            score = metric(output > 0.5, y).mean().item()
            loss_sum += loss
            score_sum += score
    return loss_sum / len(val_dl), score_sum / len(val_dl)


def run(model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        criterion,
        metric,
        epochs,
        device,
        weights_path='../artifacts/weights/',
        save_name='model_name.pth',
        max_early_stopping=3):
    if not os.path.exists(weights_path):
        os.makedirs(weights_path)
        print(f"{weights_path} created successfully")

    save_path = os.path.join(weights_path, save_name)

    best_val_loss = 1e3
    last_train_loss = 0
    last_val_loss = 1e3
    early_stopping_flag = 0
    best_state_dict = model.state_dict()
    for epoch in range(1, epochs + 1):
        print(f'Epoch #{epoch}')

        # <<<<< TRAIN >>>>>
        train_loss, train_score = train_epoch(model, train_loader,
                                              criterion, metric,
                                              optimizer, scheduler, device)
        print('      Score    |    Loss')
        print(f'Train: {train_score:.6f} | {train_loss:.6f}')

        # <<<<< EVAL >>>>>
        val_loss, val_score = eval_epoch(model, val_loader,
                                         criterion, metric, device)
        print(f'Val: {val_score:.6f} | {val_loss:.6f}', end='\n\n')
        metrics = {'train_score': train_score,
                   'train_loss': train_loss,
                   'val_score': val_score,
                   'val_loss': val_loss,
                   'lr': scheduler.get_last_lr()[-1]}

        # saving best weights by loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = model.state_dict()
            torch.save(best_state_dict, save_path)

        # weapon counter over-fitting
        if train_loss < last_train_loss and val_loss > last_val_loss:
            early_stopping_flag += 1
        if early_stopping_flag == max_early_stopping:
            print('<<< EarlyStopping >>>')
            break

        last_train_loss = train_loss
        last_val_loss = val_loss

    # loading best weights
    model.load_state_dict(best_state_dict)
    return model
