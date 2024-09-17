import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import os
from pathlib import Path
from sklearn.metrics import f1_score

from experiments.models import *
from utils.setup_funcs import PROJECT_ROOT


np.set_printoptions(linewidth=np.nan)


def train(model,loss_fn,optimizer,log_name,epochs,ese,device,
          train_loader,val_loader,logger,lr_scheduler,log_freq):

    # start tensorboard session
    writer = SummaryWriter(os.path.join(PROJECT_ROOT,"saved_data/runs",log_name)+"_"+str(time.time()))

    # log training parameters
    print("===========================================")
    for k,v in zip(locals().keys(),locals().values()):
        writer.add_text(f"locals/{k}", f"{v}")
        logger.info(f"locals/{k} --> {v}")
    print("===========================================")


    # ================== training loop ==================
    model.train()
    model = model.to(device)
    batch_iter = 0
    num_epochs_worse = 0
    checkpoint_path = os.path.join(PROJECT_ROOT,"saved_data/checkpoints",log_name) + ".pth"
    path_items = log_name.split("/")
    if  len(path_items) > 1:
        Path(os.path.join(PROJECT_ROOT,"saved_data/checkpoints",*path_items[:-1])).mkdir(parents=True, exist_ok=True)
    best_val_f1 = 0.0

    logger.info(f"****************************************** Training Started ******************************************")

    for e in range(epochs):
        model.train()
        model = model.to(device)
        if num_epochs_worse == ese:
            break

        for batch_idx, (data,target) in enumerate(train_loader):
            # stop training, run on the test set
            if num_epochs_worse == ese:
                break

            # generic batch processing
            data,target = data.to(device),target.to(device)

            # forward pass
            output = model(data)

            # loss
            train_loss = loss_fn(output,target)
            writer.add_scalar(f"train_metric/loss", train_loss, batch_iter)

            # backward pass
            train_loss.backward()

            # step
            optimizer.step()
            optimizer.zero_grad()

            # logging
            if batch_idx % log_freq == 0:
                if (100.0 * (batch_idx+1) / len(train_loader)) == 100:
                    logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] train loss: {:.3f}'.format(
                                e, len(train_loader.dataset), len(train_loader.dataset),
                                100.0 * (batch_idx+1) / len(train_loader), train_loss))
                else:
                    logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] train loss: {:.3f}'.format(
                                e, (batch_idx+1) * train_loader.batch_size, len(train_loader.dataset),
                                100.0 * (batch_idx+1) / len(train_loader), train_loss))
            batch_iter += 1

        # at end of epoch evaluate on the validation set
        val_acc,val_f1,val_loss = validate(model, val_loader, device, loss_fn)
        writer.add_scalar(f"val_metric/val_loss", val_loss, e)
        writer.add_scalar(f"val_metric/val_acc", val_acc, e)
        writer.add_scalar(f"val_metric/val_f1", val_f1, e)

        # logging
        logger.info('Train Epoch: {}, val_acc: {:.3f}, val_f1: {:.3f}, val loss: {:.3f}'.format(e,val_acc, val_f1, val_loss))

        # check if to save new chckpoint
        if best_val_f1 < val_f1:
            logger.info("==================== best validation metric ====================")
            logger.info('Train Epoch: {}, val_acc: {:.3f}, val_f1: {:.3f}, val loss: {:.3f}'.format(e,val_acc, val_f1, val_loss))
            best_val_f1 = val_f1

            torch.save({
                'epoch': e + 1,
                'model_state_dict': model.state_dict(),
                f"val_f1": val_f1,
                'val_loss': val_loss,
            }, checkpoint_path)
            num_epochs_worse = 0
        else:
            logger.info(f"info: {num_epochs_worse} num epochs without improving")
            num_epochs_worse += 1

        # check for early stopping
        if num_epochs_worse == ese:
            logger.info(f"Stopping training because validation metric did not improve after {num_epochs_worse} epochs")
            break

        if lr_scheduler is not None:
            lr_scheduler.step()

    logger.info(f"Best val f1: {best_val_f1}")
    model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
    

    logger.info("========================= Training Finished =========================")

def validate(model, val_loader, device, loss_fn,nd=None):
    model.eval()
    model = model.to(device)

    val_loss = 0

    # collect all labels and predictions, then feed to val metric specific function
    with torch.no_grad():
        predictions = []
        labels = []
        outputs = []

        for batch_idx, (data,target) in enumerate(val_loader):
            # parse the batch and send to device
            data,target = data.to(device),target.to(device)

            # get model output
            out = model(data.float())

            # get the loss
            val_loss += loss_fn(out, target)

            # parse the output for the prediction
            prediction = out.argmax(dim=1).to('cpu')

            predictions.append(prediction)
            labels.append(target.to('cpu'))
            outputs.append(out.to('cpu'))
        
        predictions = torch.cat(predictions).numpy()
        labels = torch.cat(labels).numpy()
        outputs = torch.cat(outputs)

        if nd is not None:
            sm = F.softmax(outputs,dim=1)
            val_nd = nd(sm)

        val_loss /= (len(val_loader))
        val_acc = (predictions == labels).mean()
        val_f1 = f1_score(labels,predictions,average='macro')

        if nd is not None:
            return val_acc, val_f1, val_loss, val_nd
        else:
            return val_acc, val_f1, val_loss
        
def validate_ens(models, val_loader, device, loss_fn):
    for model_i,model in enumerate(models):
        models[model_i].eval()
        models[model_i] = model.to(device)

    val_loss = 0

    # collect all labels and predictions, then feed to val metric specific function
    with torch.no_grad():
        predictions = []
        labels = []
        outputs = []

        for batch_idx, (data,target) in enumerate(val_loader):
            # parse the batch and send to device
            data,target = data.to(device),target.to(device)

            # get model output
            outs = []
            for model in models:
                outs.append(model(data.float()))
            out = torch.stack(outs).mean(dim=0)

            # get the loss
            val_loss += loss_fn(out, target)

            # parse the output for the prediction
            prediction = out.argmax(dim=1).to('cpu')

            predictions.append(prediction)
            labels.append(target.to('cpu'))
            outputs.append(out.to('cpu'))
        
        predictions = torch.cat(predictions).numpy()
        labels = torch.cat(labels).numpy()
        outputs = torch.cat(outputs)

        val_loss /= (len(val_loader))
        val_acc = (predictions == labels).mean()
        val_f1 = f1_score(labels,predictions,average='macro')

        return val_acc, val_f1, val_loss