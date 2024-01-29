import numpy as np
import config as cfg
import model_config as mcfg
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import KUL_dataset, get_EEG_emb_feat, get_stimulus_feat, get_EEG_emb_feat_test
from torch.utils.tensorboard import SummaryWriter
import logging
import argparse
import time
import copy
import os
import os.path as op
import json
from model import BrainNetworkCL



def valid_model(model, dataloaders, device, optimizer, phase='valid'):

    model.eval()  # Set model to evaluate mode

    epoch_induce = None
    sub_ids = []
    sti_ids = []

    sti_valid = dataloaders['valid'].dataset.Stimulus_index_dict
    sti_valid = np.concatenate(list(sti_valid.values()))
    sti_valid = np.unique(sti_valid).tolist()

    sti_train = dataloaders['train'].dataset.Stimulus_index_dict
    sti_train = np.concatenate(list(sti_train.values()))
    sti_train = np.unique(sti_train).tolist()

    sti_valid_all = dataloaders['valid_all'].dataset.Stimulus_index_dict
    sti_valid_all = np.concatenate(list(sti_valid_all.values()))
    sti_valid_all = np.unique(sti_valid_all).tolist()

    feat_keys = list(dataloaders['valid'].dataset.feat_keys)
    sti_out_feat_name = [feat_keys[idx] for idx in sti_valid]
    sti_in_feat_name = [feat_keys[idx] for idx in sti_valid_all if idx not in sti_valid]
    # Iterate over data.
    for iter, (eeg, emb_id, sub_id, sti_id) in enumerate(tqdm(dataloaders[phase]), start=1):
        embed = mcfg.embeds[emb_id]
        eeg, embed, sub_id = \
            eeg.to(device), embed.to(device), sub_id.to(device)
        optimizer.zero_grad()

        # forward
        with torch.no_grad():
            pred_induce = model.test(x=eeg, sub_id=sub_id, frame_id=emb_id, embed=embed, sti_id=sti_id)

        if epoch_induce is None:
            epoch_induce = pred_induce
        else:
            epoch_induce = torch.cat((epoch_induce, pred_induce))
        sub_ids.append(sub_id)
        sti_ids.append(sti_id)
    sub_ids = torch.cat(sub_ids)
    sub_ids_set = torch.unique(sub_ids).detach().cpu().numpy().tolist()
    sub_names = dataloaders['valid'].dataset.sub_names
    acc = epoch_induce == 0
    acc_sub_dict = {}
    for sub_id in sub_ids_set:
        sub_name = sub_names[sub_id]
        acc_sub = acc[sub_ids == sub_id]
        acc_sub = acc_sub.sum().item()/len(acc_sub)
        acc_sub_dict[sub_name] = acc_sub
    if phase == 'valid_all':
        sti_ids = torch.cat(sti_ids)
        sti_ids_set = torch.unique(sti_ids).detach().cpu().numpy().tolist()

        acc_sti_in = []
        acc_sti_out = []
        for sti_id in sti_ids_set:
            acc_sti = acc[sti_ids == sti_id]
            acc_sti = acc_sti.sum().item()/len(acc_sti)
            if sti_id in sti_valid:
                acc_sti_out.append(acc_sti)
            else:
                acc_sti_in.append(acc_sti)
        if len(acc_sti_in) > 0:
            acc_sti_in = np.mean(acc_sti_in)
        else:
            acc_sti_in = 0
        if len(acc_sti_out) > 0:
            acc_sti_out = np.mean(acc_sti_out)
        else:
            acc_sti_out = 0
    else:
        acc_sti_in = 0
        acc_sti_out = 0

    acc_sub_mean = np.mean(list(acc_sub_dict.values()))
    return acc_sub_mean, acc_sub_dict, acc_sti_in, acc_sti_out, model


def train_model(dataloaders, dataset_sizes, device, model,
                optimizer, scheduler, start_epoch, num_epochs,
                best_top1, best_top1_all,
                checkpoint_path_best,
                checkpoint_path_best_all,
                early_stop_num,
                logger,
                writer):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())

    if best_top1 is None:
        best_top1 = 0
        best_epoch = -1
    else:
        best_epoch = start_epoch - 1

    if best_top1_all is None:
        best_top1_all = 0
        best_epoch_all = -1
    else:
        best_epoch_all = start_epoch - 1

    global_step = 0
    iter50_loss = 0.0
    iter50_rank = None
    phase = 'train'
    un_update_step = 0
    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        model.train()  # Set model to training mode
        running_loss = 0.0
        epoch_rank = None
        # Iterate over data.
        for iter, (eeg, emb_id, sub_id, sti_id) in enumerate(tqdm(dataloaders[phase]), start=1):
            embed = mcfg.embeds[emb_id]
            eeg, embed, sub_id = \
                eeg.to(device), embed.to(device), sub_id.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            pred, loss, rank = model(x=eeg, sub_id=sub_id, frame_id=emb_id, embed=embed, sti_id=sti_id)
            loss = loss.mean(0, keepdim=True)
            if iter50_rank is None:
                iter50_rank = rank
            else:
                iter50_rank = torch.cat((iter50_rank, rank))
            if epoch_rank is None:
                epoch_rank = rank
            else:
                epoch_rank = torch.cat((epoch_rank, rank))

            iter50_loss += loss.item()
            running_loss += loss.item() * eeg.size()[0]  # batch loss
            loss.backward()
            optimizer.step()
            global_step += 1
            it_num = 1000
            if global_step % it_num == 0:
                # saving iter_50 result
                top10_iter50 = torch.nonzero(iter50_rank <= 10).shape[0] / iter50_rank.shape[0]
                top5_iter50 = torch.nonzero(iter50_rank <= 5).shape[0] / iter50_rank.shape[0]
                top1_iter50 = torch.nonzero(iter50_rank == 1).shape[0] / iter50_rank.shape[0]
                sample_size = mcfg.negsample_num + 1
                rank_acc_iter50 = (sample_size - torch.mean(iter50_rank.float())) / (sample_size - 1)
                writer.add_scalar('Iter50/Loss', iter50_loss / it_num, global_step // it_num)
                writer.add_scalars('Iter50/Accuracy',
                                   {'top-10': top10_iter50,
                                    'top-5': top5_iter50,
                                    'top-1': top1_iter50,
                                    'rank-acc': rank_acc_iter50},
                                   global_step // it_num)
                logger.info("Train: epoch {:.0f}/ global_step {:.0f}: "
                            "loss: {:.4f}; "
                            "top-10: {:.4f}; "
                            "top-5: {:.4f}; "
                            "top-1: {:.4f}; "
                            "rank-acc: {:.4f}"
                            .format(epoch, global_step,
                                    iter50_loss / it_num,
                                    top10_iter50,
                                    top5_iter50,
                                    top1_iter50,
                                    rank_acc_iter50))
                print('{} Loss: {:.4f}; top-10: {:.4f}; rank-acc: {:.4f}'
                      .format(phase, iter50_loss / it_num, top10_iter50, rank_acc_iter50))
                iter50_loss = 0.0
                iter50_rank = None

                print('evaluating valid set...')

                top1_iter50_valid, acc_sub_dict, _, _, model = \
                    valid_model(model, dataloaders, device, optimizer, phase='valid')
                top1_iter50_valid_all, acc_sub_dict_all, acc_sti_in, acc_sti_out, model = \
                    valid_model(model, dataloaders, device, optimizer, phase='valid_all')

                model.train()


                writer.add_scalars(f'Iter50/valid-Accuracy',
                                   {'top-1': top1_iter50_valid,
                                    'top-1-all': top1_iter50_valid_all},
                                   global_step // it_num)
                writer.add_scalars(f'Iter50/valid-Accuracy-sub',
                                    acc_sub_dict,
                                    global_step // it_num)
                writer.add_scalars(f'Iter50/valid-Accuracy-sub-all',
                                    acc_sub_dict_all,
                                    global_step // it_num)
                writer.add_scalars(f'Iter50/valid-Accuracy-sti',
                                   {'acc_sti_in': acc_sti_in,
                                    'acc_sti_out': acc_sti_out},
                                    global_step // it_num)
                logger.info("Dev: epoch {:.0f}/ global_step {:.0f}: "
                            "top-1: {:.4f}; "
                            "top-1-all: {:.4f}; "
                            .format(epoch, global_step, top1_iter50_valid, top1_iter50_valid_all))
                acc_sub_dict_as_str = ', '.join([f'{k}: {v:.4f}' for k, v in acc_sub_dict.items()])
                acc_sub_dict_all_as_str = ', '.join([f'{k}: {v:.4f}' for k, v in acc_sub_dict_all.items()])
                logger.info(f"Dev: epoch {epoch}/ global_step {global_step}: acc_sub_dict: {acc_sub_dict_as_str}")
                logger.info(f"Dev: epoch {epoch}/ global_step {global_step}: acc_sub_dict_all: {acc_sub_dict_all_as_str}")
                logger.info(f"Dev: epoch {epoch}/ global_step {global_step}: acc_sti_in: {acc_sti_in:.4f}")
                logger.info(f"Dev: epoch {epoch}/ global_step {global_step}: acc_sti_out: {acc_sti_out:.4f}")


                print('valid top-1: {:.4f}, top-1-all: {:.4f}'
                      .format(top1_iter50_valid, top1_iter50_valid_all))

                if  top1_iter50_valid > best_top1:
                    best_top1 = top1_iter50_valid
                    best_epoch = epoch
                    un_update_step = 0
                    # best_model_wts = copy.deepcopy(model.state_dict())
                    checkpoint = {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dic": optimizer.state_dict(),
                        "epoch": best_epoch,
                        "top1_acc": best_top1,
                        "global_step": global_step}
                    '''save checkpoint'''
                    torch.save(checkpoint, checkpoint_path_best)
                    print(f'update best on valid checkpoint: {checkpoint_path_best}')
                    logger.info(f'update best on valid checkpoint: {checkpoint_path_best}')
                else:
                    un_update_step += 1


                if  top1_iter50_valid_all > best_top1_all:
                    best_top1_all = top1_iter50_valid_all
                    best_epoch_all = epoch
                    # best_model_wts = copy.deepcopy(model.state_dict())
                    checkpoint = {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dic": optimizer.state_dict(),
                        "epoch": best_epoch_all,
                        "top1_acc": best_top1_all,
                        "global_step": global_step}
                    '''save checkpoint'''
                    torch.save(checkpoint, checkpoint_path_best_all)
                    print(f'update best on valid checkpoint: {checkpoint_path_best_all}')
                    logger.info(f'update best on valid checkpoint: {checkpoint_path_best_all}')
                logger.info("\n\n")


        scheduler.step()
        epoch_loss = running_loss / dataset_sizes[phase]

        top10_epoch = torch.nonzero(epoch_rank <= 10).shape[0] / epoch_rank.shape[0]
        top5_epoch = torch.nonzero(epoch_rank <= 5).shape[0] / epoch_rank.shape[0]
        top1_epoch = torch.nonzero(epoch_rank == 1).shape[0] / epoch_rank.shape[0]
        # epoch_rank long tensor to float tensor
        sample_size = mcfg.negsample_num + 1
        rank_acc_epoch = (sample_size - torch.mean(epoch_rank.float()))/(sample_size-1)

        writer.add_scalar(f'Loss/{phase}', epoch_loss, epoch)
        writer.add_scalars(f'Accuracy/{phase}',
                           {'top-10': top10_epoch,
                            'top-5': top5_epoch,
                            'top-1': top1_epoch,
                            'rank-acc': rank_acc_epoch},
                           epoch)
        logger.info(" epoch {:.0f}: loss/{}: {:.5f}; "
                    "top-10: {:.4f}; "
                    "top-5: {:.4f}; "
                    "top-1: {:.4f}; "
                    "rank-acc: {:.4f}"
                    .format(epoch, phase, epoch_loss, top10_epoch, top5_epoch, top1_epoch, rank_acc_epoch))
        print('{} Loss: {:.4f}; top-10: {:.4f}; rank-acc: {:.4f}'
              .format(phase, epoch_loss, top10_epoch, rank_acc_epoch))

        if un_update_step > early_stop_num:
            print(f'early stop at epoch {epoch}, global_step {global_step}')
            logger.info(f'early stop at epoch {epoch}, global_step {global_step}')
            break


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val acc epoch {:0f}: {:4f}'.format(best_epoch, best_top1))
    print('Best val for all test data acc epoch {:0f}: {:4f}'.format(best_epoch_all, best_top1_all))
    logger.info('Best val loss epoch {:0f}: {:4f}'.format(best_epoch, best_top1))
    logger.info('Best val for all test data loss epoch {:0f}: {:4f}'.format(best_epoch_all, best_top1_all))

    model.load_state_dict(best_model_wts)
    return model


def train(args):
    num_epochs = args.num_epochs
    lr = args.lr
    batch_size = args.batch_size
    model_name = args.model_name
    device = args.device
    valid_batch = 32
    opt = args.opt
    fs = args.fs
    seg_len = args.seg_len
    negsample_num = args.negsample_num
    kernel_size = args.kernel_size
    feature_name = args.feature_name
    valid = args.valid
    att_out_dim = args.att_out_dim
    use_multi_band = args.use_multi_band
    dropout = args.dropout
    early_stop_num = args.early_stop_num
    args_dict = vars(args)
    args_json = json.dumps(args_dict, indent=4)

    tag_nn = f'-nn{negsample_num}'
    tag_att = f'-att{att_out_dim}'
    tag_dropout = f'-dor{dropout}'
    tag_mbd = '-mbd' if use_multi_band else ''

    if '_' not in feature_name:
        feature_name_ls = [feature_name]
    else:
        feature_name_ls = feature_name.split('_')

    feature_name_ls_new = []
    for fn in feature_name_ls:
        if fn == 'wav2vec11layers':
            feature_name_new = [f'wav2vec{layer}pca32' for layer in range(2, 24, 2)]
            feature_name_ls_new.extend(feature_name_new)
        else:
            feature_name_ls_new.append(fn)
    feature_name_ls = feature_name_ls_new

    out_dim_ls = [mcfg.feature_dim_dict[feature_name] for feature_name in feature_name_ls]
    out_dim = sum(out_dim_ls)

    tag_emb = feature_name
    tag_valid = f'-valid{valid}'

    model_dir = f'{model_name}-bs{batch_size}-' \
                f'sl{seg_len}{tag_mbd}-ks{kernel_size}{tag_dropout}{tag_att}{tag_nn}{tag_valid}-{tag_emb}'
    print(f'model dir: {model_dir}')
    checkpoint_dir = op.join(cfg.project_dir, 'result',
                             'models', model_dir)
    checkpoint_best_dir = op.join(checkpoint_dir, 'best')
    output_checkpoint_name_best = op.join(checkpoint_best_dir, 'model.pt')
    output_checkpoint_name_best_all = op.join(checkpoint_best_dir, 'model_all.pt')
    if not op.exists(checkpoint_best_dir):
        os.makedirs(checkpoint_best_dir)
    tensorboard_dir = op.join(checkpoint_dir, 'runs')
    if not op.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    '''setting up logger file'''
    file_handler = logging.FileHandler(op.join(checkpoint_dir, 'acc_log.txt'))
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler.setFormatter(formatter)
    logger = logging.getLogger(model_dir)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.info('\n\n')
    logger.info('*****************************START NEW SESSION*****************************\n')
    logger.info('PARAMETER ...')
    logger.info(args_json)

    ''' set random seeds '''
    # seed_val = 1024
    # logger.info(f'seed_val: {seed_val}')
    # np.random.seed(seed_val)
    # torch.manual_seed(seed_val)
    # torch.cuda.manual_seed_all(seed_val)
    ''' set up device '''
    if torch.cuda.is_available():
        pass
    else:
        device = "cpu"
    print(f'[INFO]using device {device}')

    ''' load data '''
    feat_dict_ls = []
    feat_frames_ls = []
    for feat in feature_name_ls:
        feat_dict_1, feat_frames_id_dict_1, feat_keys_1, feat_frames_1 = (
            get_stimulus_feat(feature_name=feat, seg_len=seg_len, fs=fs))
        feat_dict_ls.append(feat_dict_1)
        feat_frames_ls.append(feat_frames_1)

    feat_dict = {}
    for key in feat_keys_1:
        feat_dict[key] = np.concatenate([feat_dict_1[key] for feat_dict_1 in feat_dict_ls], axis=-1)
    feat_frames = np.concatenate(feat_frames_ls, axis=-1)
    feat_frames_id_dict = feat_frames_id_dict_1
    feat_keys = feat_keys_1

    sub_names = ['sub-{:03d}'.format(i) for i in range(1, 86)]
    if valid == 0:
        sub_ls_test = [i for i in range(1, 27)]
        sub_ls_train = [i for i in range(27, 86)]
    elif valid < 5:
        sub_ls_test = np.arange(valid * 17 + 1, (valid + 1) * 17+1)
        sub_ls_test = sub_ls_test.tolist()
        sub_ls_train = [i for i in range(1, 86) if i not in sub_ls_test]
    else:
        # random select 17 subjects as test set, 68 as train set
        start1 = np.random.choice(np.arange(1, 86 - 8), 1, replace=False)[0]
        start2 = np.random.choice(np.arange(1, 86 - 8), 1, replace=False)[0]
        test1 = [i for i in range(start1, start1 + 8)]
        test2 = [i for i in range(start2, start2 + 8)]
        sub_ls_test = np.array(test1 + test2)
        sub_ls_test = np.unique(sub_ls_test)
        sub_ls_test = sub_ls_test.tolist()
        sub_ls_train = [i for i in range(1, 86) if i not in sub_ls_test]


    print(f'sub_ls_test: {sub_ls_test}')
    print(f'sub_ls_train: {sub_ls_train}')

    logger.info(f'sub_ls_test: {sub_ls_test}')
    logger.info(f'sub_ls_train: {sub_ls_train}')

    '''
    get EEG for trainset
    '''
    (EEG_dict, EEG_feat_index_dict, Sub_id_dict,
     Stimulus_index_dict, id2dict, train_stimulus_filename) = get_EEG_emb_feat(sub_ls_train,
                                                                               feat_dict,
                                                                               feat_frames_id_dict,
                                                                               feat_keys,
                                                                               seg_len=seg_len,
                                                                               phase='train')
    train_stimulus_filename = list(set(train_stimulus_filename))
    train_set = KUL_dataset(EEG_dict, EEG_feat_index_dict, Sub_id_dict,
                            Stimulus_index_dict,
                            id2dict, sub_names, feat_keys,
                            use_multi_band=use_multi_band)

    '''
    get EEG for testset (remove stimulus that exists in trainset)
    '''

    (EEG_dict, EEG_feat_index_dict, Sub_id_dict,
     Stimulus_index_dict,id2dict) = get_EEG_emb_feat_test(sub_ls_test,
                                                          feat_dict,
                                                          feat_frames_id_dict,
                                                          feat_keys,
                                                          seg_len=seg_len,
                                                          train_stimulus_filename=train_stimulus_filename)
    test_set = KUL_dataset(EEG_dict, EEG_feat_index_dict, Sub_id_dict,
                            Stimulus_index_dict,
                            id2dict, sub_names, feat_keys,
                           use_multi_band=use_multi_band)
    '''
    get EEG for testset (include stimulus that exists in trainset)
    '''
    (EEG_dict_test_all, EEG_feat_index_dict_test_all, Sub_id_dict_test_all,
     Stimulus_index_dict_test_all, id2dict_test_all, train_stimulus_filename_test_all) = get_EEG_emb_feat(sub_ls_test,
                                                                               feat_dict,
                                                                               feat_frames_id_dict,
                                                                               feat_keys,
                                                                               seg_len=seg_len,
                                                                               phase='test')
    test_set_all = KUL_dataset(EEG_dict_test_all, EEG_feat_index_dict_test_all, Sub_id_dict_test_all,
                            Stimulus_index_dict_test_all,
                            id2dict_test_all, sub_names, feat_keys,
                            use_multi_band=use_multi_band)

    feat_frames_id = [f for f in feat_frames_id_dict.values()]
    embeds = torch.tensor(feat_frames, dtype=torch.float32, device='cpu')
    mcfg.embeds = embeds
    mcfg.feat_frames_id = feat_frames_id

    dataset_sizes = {'train': len(train_set), 'test': len(test_set)}
    print('[INFO]train_set size: ', len(train_set))
    print('[INFO]valid_set size: ', len(test_set))

    # train dataloader
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=16)
    # valid dataloader
    val_dataloader = DataLoader(test_set, batch_size=valid_batch, shuffle=False, num_workers=16)
    # valid_all dataloader
    val_dataloader_all = DataLoader(test_set_all, batch_size=valid_batch, shuffle=False, num_workers=16)
    # dataloaders
    dataloaders = {'train': train_dataloader, 'valid': val_dataloader, 'valid_all': val_dataloader_all}

    ''' set up model '''
    in_dim = 64
    if use_multi_band:
        in_dim = 4*in_dim


    if model_name == 'BrainNetworkCL':
        model = BrainNetworkCL(in_dim=in_dim,
                               att_out_dim=att_out_dim,
                               out_dim=out_dim,
                               dropout=dropout,
                               train_subs=sub_ls_train,
                               negsample_num=negsample_num,
                               device=device,
                               kernel_size=kernel_size)
    else:
        raise ValueError('model_name not supported')
    model.to(device)
    ''' set up tensorboard'''
    writer = SummaryWriter(tensorboard_dir)
    ''' set up optimizer and scheduler'''
    if opt == 'Adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    ''' train model'''
    print('=== start training ... ===')
    best_top1 = None
    best_top1_all = None
    start_epoch = 0
    logger.info(f'training from the beginning, random initialized')
    print(f'training from the beginning, random initialized')
    model = train_model(dataloaders, dataset_sizes, device, model,
                optimizer=optimizer, scheduler=exp_lr_scheduler,
                start_epoch=start_epoch, num_epochs=num_epochs,
                best_top1=best_top1, best_top1_all=best_top1_all,
                checkpoint_path_best=output_checkpoint_name_best,
                checkpoint_path_best_all=output_checkpoint_name_best_all,
                early_stop_num=early_stop_num,
                logger=logger,
                writer=writer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default=mcfg.model_name)
    parser.add_argument('--seg_len', type=int, default=mcfg.seg_len)
    parser.add_argument('--fs', type=float, default=mcfg.fs)
    parser.add_argument('--device', default=mcfg.device)
    parser.add_argument('--opt', default=mcfg.opt)  # SGD, adm
    parser.add_argument('--lr', type=float, default=mcfg.lr_dic[mcfg.opt])
    parser.add_argument('--num_epochs', type=int, default=mcfg.num_epochs)
    parser.add_argument('--batch_size', type=int, default=mcfg.batch_size)
    parser.add_argument('--negsample_num', type=int, default=mcfg.negsample_num)
    parser.add_argument('--kernel_size', type=int, default=mcfg.kernel_size)
    parser.add_argument('--feature_name', default=mcfg.feature_name)
    parser.add_argument('--valid', type=int, default=mcfg.valid)
    parser.add_argument('--att_out_dim', type=int, default=mcfg.att_out_dim)
    parser.add_argument('--dropout', type=float, default=mcfg.dropout)
    parser.add_argument('--use_multi_band', type=bool, default=mcfg.use_multi_band)
    parser.add_argument('--early_stop_num', type=int, default=mcfg.early_stop_num)
    args = parser.parse_args()
    args.lr = mcfg.lr_dic[args.opt]
    train(args)
