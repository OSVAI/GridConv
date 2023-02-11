import torch
import time
from progress.bar import Bar
import numpy as np

from tool import util

def train(epoch, train_loader, lifting_model, criterion, optimizer, opt):
    losses = util.AverageMeter()

    start = time.time()
    batch_time = 0
    train_loader_len = len(train_loader)
    bar = Bar('>>>', fill='>', max=train_loader_len)

    lifting_model.train()

    for i, (inputs, targets, _) in enumerate(train_loader):
        batch_size = targets.shape[0]

        if hasattr(lifting_model, "net_update_temperature") and epoch < opt.temp_epoch:
            temperature = util.get_temperature(0, epoch, opt.temp_epoch, i, train_loader_len,
                                          method='linear', max_temp=opt.max_temp, increase=False)
            lifting_model.net_update_temperature(temperature)

        inputs_gpu = inputs.cuda()
        targets_gpu = targets.cuda()

        outputs_gpu = lifting_model(inputs_gpu)
        optimizer.zero_grad()
        loss = criterion(outputs_gpu, targets_gpu)
        losses.update(loss.item(), batch_size)
        loss.backward()

        optimizer.step()

        # update summary
        if (i + 1) % 100 == 0:
            batch_time = time.time() - start
            start = time.time()

        bar.suffix = '({batch}/{size}) | batch: {batchtime:.1}ms | Total: {ttl} | ETA: {eta:} | loss: {loss:.6f}' \
            .format(batch=i + 1,
                    size=len(train_loader),
                    batchtime=batch_time * 10.0,
                    ttl=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg)
        bar.next()
    bar.finish()
    return losses.avg

def evaluate(test_loader, lifting_model, criterion, opt):
    loss_test, outputs_array, targets_array, corresponding_info = inference(test_loader, lifting_model, criterion, opt)
    num_sample = len(outputs_array)

    outputs_array_by_action = util.rearrange_by_key(outputs_array, corresponding_info)
    targets_array_by_action = util.rearrange_by_key(targets_array, corresponding_info)

    err_ttl, err_act, err_dim = evaluate_actionwise(outputs_array_by_action, targets_array_by_action, opt.procrustes)

    print(">>> error mean of %d samples: %.3f <<<" % (num_sample, err_ttl))
    print(">>> error by dim: x: %.3f,  y:%.3f, z:%.3f <<<" % (tuple(err_dim)))
    return loss_test, err_ttl, err_dim

def inference(test_loader, lifting_model, criterion, opt):
    print('Inferring...')
    losses = util.AverageMeter()
    lifting_model.eval()
    outputs_array = []
    targets_array = []
    corresponding_info = []

    start = time.time()
    batch_time = 0
    bar = Bar('>>>', fill='>', max=len(test_loader))

    with torch.no_grad():
        for i, (inputs, targets, meta) in enumerate(test_loader):
            batch_size = targets.shape[0]
            inputs_gpu = inputs.cuda()
            info = meta
            info['fullaction'] = meta['action']
            info['action'] = list(map(lambda x: x.split(' ')[0], meta['action']))
            outputs_gpu = lifting_model(inputs_gpu)
            targets_gpu = targets.cuda()

            loss = criterion(outputs_gpu, targets_gpu)
            losses.update(loss.item(), batch_size)

            if opt.prepare_grid:
                outputs_pose = util.inverse_semantic_grid_trans(outputs_gpu.cpu().data.numpy())
                targets_pose = util.inverse_semantic_grid_trans(targets.data.numpy())
            else:
                outputs_pose = outputs_gpu.cpu().data.numpy()
                targets_pose = targets.data.numpy()

            outputs_array.append(outputs_pose)
            targets_array.append(targets_pose)
            info_list = util.dict2list(info)
            corresponding_info += info_list

            bar.suffix = '({batch}/{size}) | batch: {batchtime:.1}ms | Total: {ttl} | ETA: {eta:} | loss: {loss:.6f}' \
                .format(batch=i + 1,
                        size=len(test_loader),
                        batchtime=batch_time * 10.0,
                        ttl=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg)
            bar.next()
    bar.finish()

    outputs_array = np.vstack(outputs_array)
    targets_array = np.vstack(targets_array)
    return losses.avg, outputs_array, targets_array, corresponding_info

def evaluate_actionwise(outputs_array, targets_array, procrustes):
    err_ttl = util.AverageMeter()
    err_act = {}
    err_dim = [util.AverageMeter(), util.AverageMeter(), util.AverageMeter()]

    for act in sorted(outputs_array.keys()):
        num_sample = outputs_array[act].shape[0]
        predict = outputs_array[act] * 1000.
        gt = targets_array[act] * 1000.

        if procrustes:
            pred_procrustes = []
            for i in range(num_sample):
                _, Z, T, b, c = util.get_procrustes_transformation(gt[i], predict[i], True)
                pred_procrustes.append((b * predict[i].dot(T)) + c)
            predict = np.array(pred_procrustes)

        err_act[act] = (((predict - gt) ** 2).sum(-1)**0.5).mean()
        err_ttl.update(err_act[act], 1)
        for dim_i in range(len(err_dim)):
            err = (np.abs(predict[:, :, dim_i] - gt[:, :, dim_i])).mean()
            err_dim[dim_i].update(err, 1)

    for dim_i in range(len(err_dim)):
        err_dim[dim_i] = err_dim[dim_i].avg

    return err_ttl.avg, err_act, err_dim