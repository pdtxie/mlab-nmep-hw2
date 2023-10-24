import argparse
import datetime
import json
import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.multiprocessing as mp

from torch.utils.data.distributed import DistributedSampler

from timm.utils import AverageMeter, accuracy
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset  # For custom datasets
from tqdm import tqdm
from fvcore.nn import FlopCountAnalysis, flop_count_str

from config import get_config
from data import build_loader
from models import build_model
from optimizer import build_optimizer
from utils import create_logger, load_checkpoint, save_checkpoint, graphing


# question specific values
Q5_2_LEARNING_RATES = [1e-4, 3e-4, 1e-3, 3e-3]
Q5_3_BATCH_SIZES = [128, 256, 512, 480, 1024]
Q5_4_BATCH_SIZES = [2**i for i in range(15)]


def parse_option():
    parser = argparse.ArgumentParser("Vision model training and evaluation script", add_help=False)
    parser.add_argument("--cfg", type=str, required=True, metavar="FILE", help="path to config file")
    parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs.", default=None, nargs="+")

    # easy config modification
    parser.add_argument("--batch-size", type=int, help="batch size for single GPU")
    parser.add_argument("--data-path", type=str, help="path to dataset")
    parser.add_argument("--resume", help="resume from checkpoint")
    parser.add_argument(
        "--output",
        default="output",
        type=str,
        metavar="PATH",
        help="root of output folder, the full path is <output>/<model_name>/<tag> (default: output)",
    )
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument("--throughput", action="store_true", help="Test throughput only")

    args = parser.parse_args()

    config = get_config(args)

    return args, config


# def setup(rank):
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'

    # torch.distributed.init_process_group(backend="gloo", rank=rank, world_size=4)

# def cleanup():
    # torch.distributed.destroy_process_group()

# def ddp_run(rank, model):
    # setup(rank)

    # model = model.to(rank)
    # ddp_model = nn.parallel.DistributedDataParallel(model)

    # cleanup()


def main(config):
    dataset_train, dataset_val, dataset_test, data_loader_train, data_loader_val, data_loader_test = build_loader(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(config)
    # model = model.cuda()
    model = model.to(device)

    # logger.info(str(model))

    # mp.spawn(ddp_run, args=(model,), nprocs=4)

    # param and flop counts
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    toy_input = torch.rand(1, 3, config.DATA.IMG_SIZE, config.DATA.IMG_SIZE).to(device) # for measuring flops
    flops = FlopCountAnalysis(model, toy_input)
    del toy_input

    # print("Model = %s" % str(model_without_ddp))
    n_flops = flops.total()
    logger.info(flop_count_str(flops))
    logger.info('number of params: {} M'.format(n_parameters / 1e6))
    logger.info(f'flops: {n_flops/1e6} MFLOPS')

    # Keep it simple with basic epoch scheduler
    optimizer = build_optimizer(config, model)
    criterion = torch.nn.CrossEntropyLoss()
    lr_scheduler = CosineAnnealingLR(optimizer, config.TRAIN.EPOCHS)

    max_accuracy = 0.0

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model, optimizer, lr_scheduler, logger)
        acc1, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} val images: {acc1:.1f}%")
        if config.EVAL_MODE:
            return

    logger.info("Start training")
    start_time = time.time()

    # graphs
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    epoch_times = []

    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        train_acc1, train_loss, epoch_time = train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch)
        logger.info(f" * Train Acc {train_acc1:.3f} Train Loss {train_loss:.3f}")
        logger.info(f"Accuracy of the network on the {len(dataset_train)} train images: {train_acc1:.1f}%")

        # train_acc1, _ = validate(config, data_loader_train, model)
        val_acc1, val_loss = validate(config, data_loader_val, model)
        logger.info(f" * Val Acc {val_acc1:.3f} Val Loss {val_loss:.3f}")
        logger.info(f"Accuracy of the network on the {len(dataset_val)} val images: {val_acc1:.1f}%")

        epoch_times.append(epoch_time)

        train_accs.append(train_acc1)
        val_accs.append(val_acc1)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1):
            save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger)

        max_accuracy = max(max_accuracy, val_acc1)
        logger.info(f"Max accuracy: {max_accuracy:.2f}%\n")
        lr_scheduler.step()

        log_stats = {"epoch": epoch, "n_params": n_parameters, "n_flops": n_flops,
                     "train_acc": train_acc1, "train_loss": train_loss, 
                     "val_acc": val_acc1, "val_loss": val_loss}
        with open(os.path.join(config.OUTPUT, "metrics.json"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(total_time_str))

    logger.info("Start testing")
    preds = evaluate(config, data_loader_test, model)
    np.save(os.path.join(config.OUTPUT, "preds.npy"), preds)
    # TODO save predictions to csv in kaggle format

    # NOTE: graphing

    # INFO: q5.2
    # graphing.graph("AlexNet Validation Losses",
    #                label=("Epochs", "Validation Loss"),
    #                data=(range(len(val_losses)), val_losses),
    #                file_name=f"alexnet_val_loss_{config.TRAIN.LR}.png",
    #                legend=list(map(str, Q5_2_LEARNING_RATES)),
    #                reset=False)

    # INFO: q5.3
    # return val_accs, val_losses

    # INFO: q5.4
    return epoch_times

    # INFO: q6/7
    """
    graphing.graph("ResNet Validation Losses",
                   label=("Epochs", "Validation Loss"),
                   data=(range(len(val_losses)), val_losses),
                   file_name="resnet_val_losses.png")

    graphing.graph("ResNet Validation Accuracies",
                   label=("Epochs", "Validation Accuracy"),
                   data=(range(len(val_accs)), val_accs),
                   file_name="resnet_val_accs.png")
    """

    # plt.plot(range(len(val_losses)), val_losses)
    # plt.title("ResNet Validation Losses")

    # plt.xlabel(xlabel="Epochs")
    # plt.ylabel(ylabel="Validation Loss")

    # plt.savefig("resnet_val_losses.png")
    # plt.clf()


    # plt.plot(range(len(val_accs)), val_accs)
    # plt.title("ResNet Validation Accuracies")

    # plt.xlabel(xlabel="Epochs")
    # plt.ylabel(ylabel="Validation Accuracies")

    # plt.savefig("resnet_val_accs.png")
    # plt.clf()

    # print(val_accs)
    # print(val_losses)





def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch):
    model.train()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (samples, targets) in enumerate(tqdm(data_loader, leave=False)):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        optimizer.zero_grad()
        outputs = model(samples)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        (acc1,) = accuracy(outputs, targets)
        loss_meter.update(loss.item(), targets.size(0))
        acc1_meter.update(acc1.item(), targets.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

    lr = optimizer.param_groups[0]["lr"]
    memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    logger.info(
        f"Train: [{epoch}/{config.TRAIN.EPOCHS}]\t"
        f"lr {lr:.6f}\t"
        f"time {batch_time.val:.4f} ({batch_time.avg:.4f})\t"
        f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
        f"Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t"
        f"Mem {memory_used:.0f}MB"
    )
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    return acc1_meter.avg, loss_meter.avg, epoch_time


@torch.no_grad()
def validate(config, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(images)

        # measure accuracy and record loss
        loss = criterion(output, target)
        (acc1,) = accuracy(output, target)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    logger.info(
        f"Validate: \t"
        f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
        f"Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
        f"Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t"
        f"Mem {memory_used:.0f}MB"
    )
    return acc1_meter.avg, loss_meter.avg


@torch.no_grad()
def evaluate(config, data_loader, model):
    model.eval()
    preds = []

    for idx, (images, _) in enumerate(tqdm(data_loader)):
        images = images.cuda(non_blocking=True)
        output = model(images)
        preds.append(output.cpu().numpy())

    preds = np.concatenate(preds)

    return preds


if __name__ == "__main__":
    args, config = parse_option()

    seed = config.SEED
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    # Make output dir
    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}")

    path = os.path.join(config.OUTPUT, "config.yaml")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    config.defrost()

    # INFO: q5.2:
    # for lr in Q5_2_LEARNING_RATES:
    #     config.TRAIN.LR = lr
    #     main(config)

    # INFO: q5.3:
    # all_accs, all_losses = [], []

    # for bs in Q5_3_BATCH_SIZES:
    #     config.DATA.BATCH_SIZE = bs
    #     accs, losses = main(config)

    #     all_accs.append(accs)
    #     all_losses.append(losses)

    # print("ACCS:")
    # print(all_accs)
    # print("LOSSES:")
    # print(all_losses)

    # for acc in all_accs:
    #     graphing.graph("AlexNet Validation Accuracies",
    #                    label=("Epochs", "Validation Accuracy"),
    #                    data=(range(len(acc)), acc),
    #                    file_name="alexnet_val_acc.png",
    #                    legend=list(map(str, Q5_2_LEARNING_RATES)),
    #                    reset=False)

    # plt.clf()

    # for loss in all_losses:
    #     graphing.graph("AlexNet Validation Losses",
    #                    label=("Epochs", "Validation Loss"),
    #                    data=(range(len(loss)), loss),
    #                    file_name="alexnet_val_loss.png",
    #                    legend=list(map(str, Q5_2_LEARNING_RATES)),
    #                    reset=False)

    # plt.clf()

    # INFO: q5.4:
    all_epoch_times = []

    for i, bs in enumerate(Q5_4_BATCH_SIZES):
        config.DATA.BATCH_SIZE = bs
        print(f"TRAINING FOR BATCH SIZE {bs} = {(i+1)}/{len(Q5_4_BATCH_SIZES)}")
        epoch_times = main(config)
        avg = np.average(epoch_times)
        print("\n"*10)
        print(bs)
        print(avg)
        all_epoch_times.append(avg)

    print(all_epoch_times)

    throughputs = []
    for i, time in enumerate(all_apoch_times):
        thp = time / Q5_4_BATCH_SIZES[i]
        throughputs.append(thp)

    plt.bar(Q5_4_BATCH_SIZES, throughputs)
    plt.xlabel("Batch Size")
    plt.ylabel("Throughput")
    plt.savefig("alexnet_throughputs.png")
    plt.clf()

    # INFO: q6/7
    # main(config)
