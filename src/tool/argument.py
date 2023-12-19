import argparse
from pprint import pprint
import os

class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):
        # ===============================================================
        #                     General options
        # ===============================================================
        self.parser.add_argument('--data_rootdir',  type=str, default='./data/')
        self.parser.add_argument('--input',       type=str, default='gt', help='choises:{gt,cpn,sh}')

        self.parser.add_argument('--eval', dest='eval', action='store_true')
        self.parser.set_defaults(eval=False)
        self.parser.add_argument('--exp',           type=str, default='temporary', help='name of experiment')
        self.parser.add_argument('--ckpt',      type=str, default='checkpoint')
        self.parser.add_argument('--procrustes', dest='procrustes', action='store_true',
                                 help='use procrustes analysis at testing')

        self.parser.add_argument('--lr', type=float, default=1.0e-3)
        self.parser.add_argument('--lr_decay', type=int, default=10, help='milestone epoch for lr decay')
        self.parser.add_argument('--lr_gamma', type=float, default=0.96, help='decay weight')
        self.parser.add_argument('--epoch', type=int, default=200)
        self.parser.add_argument('--dropout', type=float, default=0.25, help='dropout probability')
        self.parser.add_argument('--batch', type=int, default=200)
        self.parser.add_argument('--test_batch', type=int, default=1000)
        self.parser.add_argument('--loss', type=str, default='l2')

        self.parser.add_argument('--max_temp', type=int, default=30)
        self.parser.add_argument('--temp_epoch', type=int, default=10)

        # ===============================================================
        #                     Model options
        # ===============================================================
        self.parser.add_argument('--lifting_model', type=str, default='gridconv', help='choices: {gridconv, dgridconv, dgridconv_autogrids}')
        self.parser.add_argument('--load', type=str, default=None)
        self.parser.add_argument('--hidsize',          type=int, default=256, help='number of hidden node in nn.linear layer')
        self.parser.add_argument('--num_block',      type=int, default=2, help='number of residual blocks')
        self.parser.add_argument('--padding_mode', type=str, nargs='+', default=['c','r'])
        self.parser.add_argument('--grid_shape', type=int, nargs='+', default=[5, 5])
        self.parser.add_argument('--autosgt_prior', type=str, default='standard')


    def _print(self):
        print("\n==================Options=================")
        pprint(vars(self.opt), indent=4)
        print("==========================================\n")

    def parse(self):
        self._initial()
        self.opt = self.parser.parse_args()
        ckpt = os.path.join(self.opt.ckpt, self.opt.exp)
        if not os.path.isdir(ckpt):
            os.makedirs(ckpt)
        self.opt.ckpt = ckpt
        self.opt.prepare_grid = self.opt.lifting_model in ['gridconv', 'dgridconv']
        self._print()

        return self.opt
