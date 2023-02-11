import torch
import os
import time
import numpy as np

from tool import log
from tool.argument import Options
from lifting import train, evaluate
from base_modules import get_dataloader, get_lifting_model, get_loss, get_optimizer

def main(opt):
    date = time.strftime("%y_%m_%d_%H_%M", time.localtime())
    log.save_options(opt, opt.ckpt)

    lifting_model = get_lifting_model(opt)

    print(">>> Loading dataset...")
    if not opt.eval:
        train_loader = get_dataloader(opt, is_train=True, shuffle=True)
    test_loader = get_dataloader(opt, is_train=False, shuffle=False)

    criterion = get_loss(opt)

    if opt.eval:
        logger = log.Logger(os.path.join(opt.ckpt, 'inference-%s.txt' % date))
        logger.set_names(['loss_test', 'err_test', 'err_x', 'err_y', 'err_z'])
        print(">>> Test lifting<<<")
        loss_test, err_test, err_dim = evaluate(test_loader, lifting_model, criterion, opt)
        logger.addmsg('lifting')
        logger.append([loss_test, err_test, err_dim[0], err_dim[1], err_dim[2]],
                      ['float', 'float', 'float', 'float', 'float'])
        return

    logger = log.Logger(os.path.join(opt.ckpt, 'log-%s.txt' % date))
    logger.set_names(['epoch', 'lr', 'loss_train', 'loss_test', 'err_test', 'err_x', 'err_y', 'err_z'])
    optimizer, scheduler = get_optimizer(lifting_model, opt)

    err_best = np.inf

    for epoch in range(opt.epoch):
        lr_now = scheduler.get_lr()[0]
        print('==========================')
        print('>>> epoch: {} | lr: {:.8f}'.format(epoch + 1, lr_now))

        loss_train = train(epoch, train_loader, lifting_model, criterion, optimizer, opt)
        loss_test, err_test, err_dim = evaluate(test_loader, lifting_model, criterion, opt)

        if epoch % opt.lr_decay == 0:
            scheduler.step()

        logger.append([epoch+1, lr_now, loss_train, loss_test, err_test, err_dim[0], err_dim[1], err_dim[2]],
                      ['int', 'float', 'float', 'float', 'float', 'float', 'float', 'float'])

        is_best = err_test < err_best
        err_best = min(err_test, err_best)

        # save ckpt
        stored_model_weight = lifting_model.state_dict()
        log.save_ckpt({'epoch': epoch+1,
                       'lr': lr_now,
                       'error': err_test,
                       'state_dict': stored_model_weight,
                       'optimizer': optimizer},
                      ckpt_path=opt.ckpt,
                      is_best=is_best)


if __name__ == '__main__':
    opt = Options().parse()
    main(opt)