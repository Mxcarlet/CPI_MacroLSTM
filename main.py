import torch as th
import torch.optim as optim
import torch.utils.data as torch_data
import random
from models.CroCoLSTM import LSTMModel as CroCoLSTM
from torch.utils.data.sampler import SequentialSampler
from dataloader import Data_Set, Loader
import datetime
from tqdm import tqdm
import argparse
import os
import sys
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import math
import pprint as ppr

global_log_file = None

def pprint(*args):
    # print with UTC+8 time
    time = '['+str(datetime.datetime.utcnow()+
                   datetime.timedelta(hours=8))[:19]+'] -'
    print(time, *args, flush=True)

    if global_log_file is None:
        return
    with open(global_log_file, 'a') as f:
        print(time, *args, flush=True, file=f)

def adjust_learning_rate(optimizer, epoch, learning_rate):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))

    # lr_adjust = {epoch: learning_rate * (0.5 ** ((epoch - 1) // 1))}
    lr_adjust = {epoch: learning_rate * 0.4}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


def mse(pred, label):
    loss = (pred - label)**2
    return th.mean(loss)

def mape(pred, label):
    diff = ((pred - label)/label).abs()
    return 100. * th.mean(diff)

def mae(pred, label):
    loss = (pred - label).abs()
    return th.mean(loss)

def rmse(pred, label):
    loss = (pred - label)**2
    return th.sqrt(th.mean(loss))

def TIC(pred, label):
    part1 = th.sqrt(th.mean((pred-label)**2))
    part2 = th.sqrt(th.mean((label)**2)) + th.sqrt(th.mean((pred)**2))
    return  part1/part2

def metric(pred, label):
    return mse(pred, label), mae(pred, label), rmse(pred, label), mape(pred, label), TIC(pred, label)

class Trainer(object):
    def __init__(self, nnet, optimizer, module,
                 checkpoint, gpuid, train_loader, val_loader, test_loader, total_epoch, resume=None):
        if not isinstance(gpuid, tuple):
            gpuid = (gpuid, )
        self.device = th.device("cuda:{}".format(gpuid[0]))
        self.gpuid = gpuid
        self.module = module

        # self.nnet = nnet
        # self.scheduler = scheduler
        self.optimizer = optimizer
        self.checkpoint = checkpoint
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.total_epoch = total_epoch
        self.cur_epoch = 0
        self.no_impr = 0
        self.best_epoch = 0
        # self.scheduler.best = 100
        self.best_score = th.inf


        if resume:
            if not os.path.exists(resume):
                raise FileNotFoundError(
                    "Could not find resume checkpoint: {}".format(resume))
            cpt = th.load(resume, map_location="cpu")
            self.cur_epoch = cpt["epoch"]
            print("Resume from checkpoint {}: epoch {:d}".format(
                resume, self.cur_epoch))

            model2_dict = nnet.state_dict()
            state_dict = {k: v for k, v in cpt["model_state_dict"].items() if k in model2_dict.keys()}
            model2_dict.update(state_dict)
            nnet.load_state_dict(model2_dict)
            # nnet.load_state_dict(cpt["model_state_dict"])
            self.nnet = nnet.to(self.device)

        else:
            self.nnet = nnet.to(self.device)


    def save_checkpoint(self, best=True):
        cpt = {
            "epoch": self.cur_epoch,
            "model_state_dict": self.nnet.state_dict(),
            "optim_state_dict": self.optimizer.state_dict()
        }
        th.save(
            cpt,
            os.path.join(self.checkpoint,
                         "{0}.pt.tar".format("best" if best else "last")))
        if not best and self.cur_epoch != None:
            th.save(
                cpt,
                os.path.join(self.checkpoint,
                             "Epoch{:d}.pt.tar".format(int(self.cur_epoch))))

    def run(self):
        with th.cuda.device(self.gpuid[0]):
            pprint("Set train mode...")
            for epoch in range(self.total_epoch):
                self.nnet.train()
                # train
                self.cur_epoch = epoch+1
                self.nnet.train()

                t_loss = 0
                for step, (driven, target, target_label) in enumerate(tqdm(self.train_loader)):
                    train_driven = driven.to(self.device)
                    train_target = target.to(self.device)
                    train_target_label = target_label.to(self.device)
                    self.optimizer.zero_grad()
                    loss,_ = self.nnet(train_driven, train_target, train_target_label)
                    t_loss += loss

                    loss.backward()
                    # th.nn.utils.clip_grad_value_(self.nnet.parameters(), 3.)
                    self.optimizer.step()
                    # pprint('Loss: {:.2f}'.format(loss.item()))
                pprint('Train EPOCH {:d}, avg Loss={:.6f} (lr={:.3e})'.format( self.cur_epoch, t_loss / (step+1),self.optimizer.param_groups[0]["lr"]))

                # eval
                t_loss = 0
                mses, maes, rmses, mapes, tics= 0,0,0,0,0
                self.nnet.eval()
                with th.no_grad():
                    for step, (driven, target, target_label) in enumerate(self.val_loader):
                        val_driven = driven.to(self.device)
                        val_target = target.to(self.device)
                        val_target_label = target_label.to(self.device)
                        loss, out = self.nnet(val_driven, val_target, val_target_label)

                        mse, mae, rmse, mape, tic= metric(out, val_target_label)
                        mses += mse
                        maes += mae
                        rmses += rmse
                        mapes += mape
                        tics += tic
                        t_loss += loss

                pprint('Eval mse={:.6f}, Eval mae={:.6f}, Eval rmse={:.6f}, Eval mape={:.2f}%, Eval TIC={:.6f}'.
                       format(mses/(step+1), maes/(step+1), rmses/(step+1), mapes/(step+1), tics/(step+1)))
                score = mses/(step+1)

                if score > self.best_score:
                    self.no_impr += 1
                    # pprint("no impr, best = {:.4f}".format(self.scheduler.best))
                else:
                    self.best_score = score
                    self.no_impr = 0
                    self.best_epoch = epoch + 1
                    self.save_checkpoint(best=True)

                # if self.module == 'CroCoLSTM':
                if (epoch + 1) % 15 == 0:adjust_learning_rate(self.optimizer, epoch + 1, self.optimizer.param_groups[0]["lr"])
                # else:
                    # self.scheduler.step(score)

                sys.stdout.flush()

                self.save_checkpoint(best=False)

                if self.no_impr == 5:
                    pprint(
                        "Stop training cause no impr for {:d} epochs".format(
                            self.no_impr))
                    break

            pprint("Training Finish! | best epoch:{:d}".format(self.best_epoch))

            # test
            cpt = th.load(self.checkpoint+'/best.pt.tar',map_location='cpu')
            self.nnet.load_state_dict(cpt["model_state_dict"])
            pprint("Load checkpoint from {}, epoch {:d}".format(self.checkpoint+'/best.pt.tar', cpt["epoch"]))
            self.nnet.eval()
            mses, maes, rmses, mapes, tics = 0, 0, 0, 0, 0
            with th.no_grad():
                for step, (driven, target, target_label) in enumerate(self.test_loader):
                    test_driven = driven.to(self.device)
                    test_target = target.to(self.device)
                    test_target_label = target_label.to(self.device)

                    _, out = self.nnet(test_driven, test_target, test_target_label)

                    mse, mae, rmse, mape, tic = metric(out, test_target_label)
                    mses += mse
                    maes += mae
                    rmses += rmse
                    mapes += mape
                    tics += tic

            pprint(
                'Test mse={:.6f}, Test mae={:.6f}, Test rmse={:.6f}, Test mape={:.2f}%, Test TIC={:.6f}'.
                    format(mses / (step + 1), maes / (step + 1), rmses / (step + 1), mapes / (step + 1),
                           tics / (step + 1)))

            f = open("result.txt", 'a')
            setting = '{}_sl{}_pl{}'.format(
                args.module,
                args.seq_len,
                args.pred_len,
                )
            f.write(setting + "  \n")
            f.write('mse:{}, mae:{}, rmse:{}, mape:{}%, TIC:{}'
                    .format(mses/(step+1), maes/(step+1), rmses/(step+1), mapes/(step+1), tics / (step + 1)))
            f.write('\n')
            f.write('\n')
            f.close()
def data_split(full_list, ratio1, ratio2, shuffle=False):

    n_total = len(full_list)
    offset1 = int(n_total * ratio1)
    offset2 = int(n_total * (ratio1 + ratio2))
    if n_total == 0 or offset1 < 1 or offset2 < 1:
        return [], [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset1]
    sublist_2 = full_list[offset1:offset2]
    sublist_3 = full_list[offset2:]
    return sublist_1, sublist_2, sublist_3

def main(args):
    global global_log_file
    global_log_file = args.checkpoint + '/' + 'run.log'
    if not os.path.exists(args.checkpoint): os.makedirs(args.checkpoint)

    driven_datas, target_datas = Loader(args.mat_path)

    split = [0.7, 0.1, 0.2]
    train_index, val_index, test_index = data_split(list(range(0, (len(driven_datas) - args.seq_len - args.pred_len + 1))),split[0], split[1])

    train_set = Data_Set(driven_datas, target_datas, train_index, args.seq_len, args.pred_len)
    train_loader = torch_data.DataLoader(train_set, batch_size=args.batch_size, drop_last=False, shuffle=True, num_workers=2)

    val_set = Data_Set(driven_datas, target_datas, val_index, args.seq_len, args.pred_len)
    val_sc = SequentialSampler(val_set)
    val_loader = torch_data.DataLoader(val_set, batch_size=args.batch_size, drop_last=False, shuffle=False, num_workers=2, sampler=val_sc)

    test_set = Data_Set(driven_datas, target_datas, test_index, args.seq_len, args.pred_len)
    test_sc = SequentialSampler(test_set)
    test_loader = torch_data.DataLoader(test_set, batch_size=args.batch_size, drop_last=False, shuffle=False, num_workers=2, sampler=test_sc)

    if args.module == "CroCoLSTM":
        model = CroCoLSTM(args)
    else:
        raise RuntimeError("please check the module name!!!")
    gpuids = tuple(map(int, args.gpus.split(",")))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = ReduceLROnPlateau(
    #     optimizer,
    #     mode="min",
    #     factor=0.5,
    #     patience=3,
    #     min_lr=1e-8,
    #     verbose=True)
    trainer = Trainer(model, optimizer, args.module,
                      args.checkpoint, gpuids, train_loader, val_loader, test_loader,args.epochs,args.resume)

    trainer.run()

if __name__ == "__main__":
    fix_seed = 2022
    random.seed(fix_seed)
    th.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(
        description="",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gpus",type=str,default="0",help="Training on which GPUs ""(one or more, egs: 0, \"0,1\")")
    parser.add_argument("--module",type=str,default="CroCoLSTM",help="Training module ")
    parser.add_argument("--epochs",type=int,default=100,help="Number of training epochs")
    parser.add_argument("--mat_path",type=str,default="./datasets/Datas.mat")
    parser.add_argument("--resume",type=str,default="",help="Exist model to resume training from")
    parser.add_argument("--checkpoint",type=str,default='./checkpoint/test',help="Directory to dump models")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seq_len', type=int, default=60, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')
    parser.add_argument('--driven_size', type=int, default=127)
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    args = parser.parse_args()
    pprint("Arguments in command:\n{}".format(ppr.pformat(vars(args))))
    main(args)
    # eval_main(args)