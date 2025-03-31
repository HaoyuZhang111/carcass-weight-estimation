import argparse
from pathlib import Path
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim.lr_scheduler import StepLR
import torch.utils.data
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pretrainedmodels
import pretrainedmodels.utils
from model import se_resnext50_32x4d
from dataset import FaceDataset
from defaults import _C as cfg
import ssl
from torch import Tensor
from math import sqrt

np.set_printoptions(threshold=np.inf)
ssl._create_default_https_context = ssl._create_unverified_context

def get_args():
    model_names = sorted(name for name in pretrainedmodels.__dict__
                         if not name.startswith("__")
                         and name.islower()
                         and callable(pretrainedmodels.__dict__[name]))
    parser = argparse.ArgumentParser(description=f"available models: {model_names}",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", type=str, required=True, help="Data root directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint if any")
    parser.add_argument("--checkpoint", type=str, default="checkpoint", help="Checkpoint directory")
    parser.add_argument("--tensorboard", type=str, default=None, help="Tensorboard log directory")
    parser.add_argument('--multi_gpu', action="store_true", help="Use multi GPUs (data parallel)")
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")
    args = parser.parse_args()
    return args

def MAPE(actual,forecast):
    APE = []
    for day in range(len(actual)):
        per_err = (actual[day] - forecast[day]) / actual[day]
        per_err = abs(per_err)
        APE.append(per_err)
    MAPE = sum(APE) / len(APE)
    print(f'''
    MAPE   : {round(MAPE, 5)}
    MAPE % : {round(MAPE * 100, 5)} %
    ''')
    return round(MAPE, 5)

class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

def train(train_loader, model, criterion, optimizer, epoch, device):
    model.train()
    loss_monitor = AverageMeter()
    accuracy_monitor = AverageMeter()

    with tqdm(train_loader) as _tqdm:
        for x1, x2, y in _tqdm:
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)

            outputs = model(x1,x2)
            outputs = outputs.flatten()
            y = Tensor.float(y)
            loss = criterion(outputs, y)
            cur_loss = loss.item()
            predicted = outputs
            correct_num = predicted.eq(y).sum().item()

            sample_num = x1.size(0)
            loss_monitor.update(cur_loss, sample_num)
            accuracy_monitor.update(correct_num, sample_num)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _tqdm.set_postfix(OrderedDict(stage="train", epoch=epoch, loss=loss_monitor.avg),
                              acc=accuracy_monitor.avg, correct=correct_num, sample_num=sample_num)

    return loss_monitor.avg, accuracy_monitor.avg


def validate(validate_loader, model, criterion, epoch, device):
    model.eval()
    loss_monitor = AverageMeter()
    accuracy_monitor = AverageMeter()
    preds = []
    gt = []

    with torch.no_grad():
        with tqdm(validate_loader) as _tqdm:
            for x1, x2, y in _tqdm:
                x1 = x1.to(device)
                x2 = x2.to(device)
                y = y.to(device)

                y = Tensor.float(y)
                outputs = model(x1,x2)
                outputs = outputs.flatten()

                preds.append(outputs.cpu().numpy())
                gt.append(y.cpu().numpy())

                if criterion is not None:
                    loss = criterion(outputs, y)
                    cur_loss = loss.item()

                    predicted = outputs
                    correct_num = predicted.eq(y).sum().item()

                    sample_num = x1.size(0)
                    loss_monitor.update(cur_loss, sample_num)
                    accuracy_monitor.update(correct_num, sample_num)
                    _tqdm.set_postfix(OrderedDict(stage="val", epoch=epoch, loss=loss_monitor.avg),
                                      acc=accuracy_monitor.avg, correct=correct_num, sample_num=sample_num)

    preds = np.concatenate(preds, axis=0)
    gt = np.concatenate(gt, axis=0)
    ave_preds = preds
    diff = ave_preds - gt
    mae = np.abs(diff).mean()
    mse = np.sum((ave_preds - gt) ** 2) / len(gt)
    rmse = sqrt(mse)
    r2 = 1 - mse / np.var(gt)
    print(" mae:", mae, "mse:", mse, " rmse:", rmse, " r2:", r2)
    mape = MAPE(gt, preds)

    return loss_monitor.avg, accuracy_monitor.avg, mae,mse,rmse,r2,mape


def main():
    args = get_args()

    if args.opts:
        cfg.merge_from_list(args.opts)

    cfg.freeze()
    start_epoch = 0
    checkpoint_dir = Path(args.checkpoint)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # create model
    print("=> creating model '{}'".format(cfg.MODEL.ARCH))
    model = se_resnext50_32x4d()

    if cfg.TRAIN.OPT == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.TRAIN.LR,
                                    momentum=cfg.TRAIN.MOMENTUM,
                                    weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR,weight_decay=0.0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    resume_path = args.resume

    if resume_path:
        if Path(resume_path).is_file():
            print("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path, map_location="cpu")
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_path, checkpoint['epoch']))
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(resume_path))

    if args.multi_gpu:
        model = nn.DataParallel(model)

    if device == "cuda":
        cudnn.benchmark = True

    criterion = nn.MSELoss().to(device)
    train_dataset = FaceDataset(args.data_dir, "train_",img_size=cfg.MODEL.IMG_SIZE, augment=True,
                                age_stddev=cfg.TRAIN.AGE_STDDEV)
    train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                              num_workers=cfg.TRAIN.WORKERS, drop_last=False)

    scheduler = StepLR(optimizer, step_size=cfg.TRAIN.LR_DECAY_STEP, gamma=cfg.TRAIN.LR_DECAY_RATE,
                       last_epoch=start_epoch - 1)
    best_mae = 10000.0
    train_writer = None

    if args.tensorboard is not None:
        opts_prefix = "_".join(args.opts)
        train_writer = SummaryWriter(log_dir=args.tensorboard + "/" + opts_prefix + "_train")

    for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
        # train
        train_loss, train_acc= train(train_loader, model, criterion, optimizer, epoch, device)
        _, _, val_mae,mse,rmse,r2,mape = validate(train_loader, model, criterion, epoch, device)

        if args.tensorboard is not None:
            train_writer.add_scalar("loss", train_loss, epoch)
            train_writer.add_scalar("acc", train_acc, epoch)


        # checkpoint
        if val_mae < best_mae:
            print(f"=> [epoch {epoch:03d}] best "
                  f"mae was improved from {best_mae:.3f} to {val_mae:.3f}")
            model_state_dict = model.module.state_dict() if args.multi_gpu else model.state_dict()
            torch.save(
                {
                    'epoch': epoch + 1,
                    'arch': cfg.MODEL.ARCH,
                    'state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict()
                },
                str(checkpoint_dir.joinpath("epoch{:03d}_{:.5f}_{:.4f}.pth".format(epoch, train_loss, val_mae)))
            )
            best_mae = val_mae
        else:
            print(f"=> [epoch {epoch:03d}] best mae was not improved from {best_mae:.3f} ({val_mae:.3f})")

        test_dataset = FaceDataset(args.data_dir, "test_", img_size=cfg.MODEL.IMG_SIZE, augment=False)
        test_loader = DataLoader(test_dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,
                                 num_workers=cfg.TRAIN.WORKERS, drop_last=False)

        print("=> start testing")
        scheduler.step()

    print("=> training finished")
    print(f"additional opts: {args.opts}")
    print(f"best val mae: {best_mae:.3f}")

if __name__ == '__main__':
    main()
