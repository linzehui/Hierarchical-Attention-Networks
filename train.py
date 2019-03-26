import torch
from tqdm import tqdm
import time
from functional import *
from config import ops
train_step = 0



def _train_batch(hier_model, batch, optimizer,
                 criterion, device, summary_writer=None):

    batch_seq, doc_n_sents, batch_label = batch
    # print('label distribution:',cal_label_dist(batch_label))
    batch_seq, batch_label = batch_seq.to(device), batch_label.to(device)
    preds, _, _ = hier_model(batch_seq, doc_n_sents)
    loss = criterion(preds, batch_label)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(hier_model.parameters(),ops.clip)
    optimizer.step()

    preds = torch.argmax(preds, dim=1)
    n_correct = torch.sum(preds == batch_label)

    # --- record --- #
    global train_step
    if summary_writer:
        summary_writer.add_scalar('Training/loss_step', loss.item(), train_step)
        summary_writer.add_scalar('Training/n_correct_step',n_correct.item(),train_step)
        train_step += 1

    return loss.item(), n_correct.item()


def _train_epoch(hier_model, dataloader, optimizer,
                 criterion, device, summary_writer=None):
    hier_model.train()
    total_loss = 0
    n_total = 0
    total_correct = 0

    for batch in tqdm(
            dataloader, mininterval=2,
            desc='   - (Training)   ', leave=False):
        loss, n_correct = _train_batch(hier_model, batch, optimizer,
                                       criterion, device, summary_writer)
        batch_size=len(batch[2])
        total_loss += loss*batch_size
        total_correct += n_correct
        n_total += batch_size

    return total_loss / n_total, total_correct / n_total


def _eval_epoch(hier_model, dataloader, criterion, device, summary_writer=None):
    hier_model.eval()

    total_loss = 0
    n_correct = 0
    n_total = 0

    with torch.no_grad():
        for batch in tqdm(
                dataloader, mininterval=2,
                desc='   - (Testing)   ', leave=False):
            batch_seq, doc_n_sents, batch_label = batch
            batch_size=len(doc_n_sents)
            batch_seq, batch_label = batch_seq.to(device), batch_label.to(device)
            preds, _, _ = hier_model(batch_seq, doc_n_sents)
            loss = criterion(preds, batch_label)
            total_loss += loss.item()*batch_size
            preds = torch.argmax(preds, dim=1)
            n_correct += torch.sum(preds == batch_label).item()
            n_total += len(batch_label)

    return total_loss / n_total, n_correct / n_total


def train(hier_model, train_loader, valid_loader,test_loader, optimizer, criterion,
          device,ops, summary_writer=None):
    best_valid_accu = 0
    best_valid_loss=1000
    epoch=ops.epoch
    cnt=0
    for e_i in range(epoch):
        print('[ Epoch', e_i, ']')
        # train
        start = time.time()
        train_loss, train_accu = _train_epoch(hier_model, train_loader,
                                              optimizer, criterion, device, summary_writer=summary_writer)
        print('(Training) loss:{loss:8.5f},accuracy:{accu:3.3f}%,'
              'elapse:{elapse:3.3f} min'.format(
            loss=train_loss, accu=100 * train_accu, elapse=(time.time() - start) / 60
        ))

        # validate
        # validate
        start = time.time()
        valid_loss, valid_accu = _eval_epoch(hier_model, valid_loader, criterion, device,summary_writer)
        print('(Validation) loss:{loss:8.5f},accuracy:{accu:3.3f}%,'
              'elapse:{elapse:3.3f} min'.format(
            loss=valid_loss, accu=100 * valid_accu, elapse=(time.time() - start) / 60
        ))
        test_loss, test_accu = _eval_epoch(hier_model, test_loader, criterion, device,summary_writer)
        print('(Test) loss:{loss:8.5f},accuracy:{accu:3.3f}%,'
              .format(
            loss=test_loss, accu=100 * test_accu
        ))

        if best_valid_accu < valid_accu:
            save_chkpt(hier_model.state_dict(), epoch=e_i, valid_accu=valid_accu)
            best_valid_accu = valid_accu

        if valid_loss>best_valid_loss:
            cnt+=1
            if cnt>ops.interval:
                print('[Info] lr decreased after valid_loss not going down for 5 '
                      'consecutive times')
                for param_group in optimizer.param_groups:  # TODO check if needed
                    param_group['lr'] = param_group['lr'] * 0.5
                cnt = 0  # clean
        else:
            cnt = 0  # clean


        if summary_writer is not None:
            summary_writer.add_scalar('Validation/accu', valid_accu, e_i)
            summary_writer.add_scalar('Validation/loss', valid_loss, e_i)
            summary_writer.add_scalar('Training/accu', train_accu, e_i)
            summary_writer.add_scalar('Training/loss', train_loss, e_i)
            summary_writer.add_scalar('Test/accu', test_accu, e_i)
            summary_writer.add_scalar('Test/loss', test_loss, e_i)