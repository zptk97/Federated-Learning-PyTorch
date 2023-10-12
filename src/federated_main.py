#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

# python federated_main.py --model=cnn --dataset=cifar --local_bs=100 --local_ep=1 --epochs 700 --gpu=cuda:0 --iid=0

import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
import torchsummary

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details

if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    # logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    logger = SummaryWriter('../outputs/runs/fed_{}_{}_round{}_C[{}]_iid[{}]_localE[{}]_localB[{}]_lr{}_decay{}'.
                           format(args.dataset, args.model, args.epochs, args.frac,
                                  args.iid, args.local_ep, args.local_bs, args.lr, args.lr_decay))

    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # torchsummary.summary(global_model, input_size=(3, 32, 32))
    # exit(0)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    test_loss, test_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    lr = args.lr

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch + 1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch, lr=lr)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        lr = lr * args.lr_decay

        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            # model=local model이 되야하지 않나? 수정해봄 -> local_model은 model이 아님 class임
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc) / len(list_acc))

        # print global training loss after every 'i' rounds
        if (epoch + 1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch + 1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100 * train_accuracy[-1]))

        # Global test inference after aggregation per communication round
        test_acc, test_loss_tmp = test_inference(args, global_model, test_dataset)
        test_loss_tmp = test_loss_tmp / 128
        print(f' \n Results after {epoch + 1} global rounds of training:')
        print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
        print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))
        print("|---- Test Loss: {}".format(test_loss_tmp))
        logger.add_scalar('test_acc', test_acc, epoch)
        logger.add_scalar('test_loss', test_loss_tmp, epoch)
        test_accuracy.append(test_acc)
        test_loss.append(test_loss_tmp)


    # Test inference after completion of training
    # test_acc, test_loss = test_inference(args, global_model, test_dataset)

    # Saving the objects train_loss and train_accuracy:
    file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'. \
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))

    # PLOTTING (optional)
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use('Agg')

    # Plot Server Test acc curve
    plt.figure()
    plt.title('Test Acc vs Communication rounds')
    plt.plot(range(len(test_accuracy)), test_accuracy, color='r')
    plt.ylabel('Test Acc')
    plt.xlabel('Communication Rounds')
    plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_test_acc.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs))

    # Plot Server Test loss curve
    plt.figure()
    plt.title('Test Loss vs Communication rounds')
    plt.plot(range(len(test_loss)), test_loss, color='r')
    plt.ylabel('Test Loss')
    plt.xlabel('Communication Rounds')
    plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_test_loss.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs))

    test_accuracy = np.array(test_accuracy)
    test_loss = np.array(test_loss)
    np.save('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_test_acc'.
            format(args.dataset, args.model, args.epochs, args.frac,
                   args.iid, args.local_ep, args.local_bs), test_accuracy)
    np.save('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_test_loss'.
            format(args.dataset, args.model, args.epochs, args.frac,
                   args.iid, args.local_ep, args.local_bs), test_loss)
    # # Plot Clients Averaged Train Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_loss)), train_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
    #
    # # Plot Clients Average Train Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
