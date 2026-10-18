import os

import torch
import random
import argparse
import numpy as np
from torch.cuda import amp
from data_loader.loader import Loader
from core import Base, train, test_vcm, test_bupt
from tools import make_dirs, Logger, time_now
import warnings
warnings.filterwarnings("ignore")
import time
from datetime import timedelta


best_rank1 = 0
def seed_torch(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(config):
    seed = config.seed

    seed_torch(seed)
    all_start_time = time.monotonic()

    global best_rank1
    global best_cmc
    global best_eval_str0, best_eval_str1
    global best_epoch

    best_cmc = 0.0
    best_eval_str0 = best_eval_str1 = None
    best_epoch = 0
    loaders = Loader(config)
    model = Base(config)

    make_dirs(model.output_path)
    make_dirs(model.save_model_path)
    make_dirs(model.save_logs_path)

    logger = Logger(os.path.join(os.path.join(config.output_path, 'logs/'), 'log.txt'))
    logger('\n' * 3)
    logger(config)

    logger("Model size: {:.5f}M".format(sum(p.numel() for p in model.model.parameters()) / 1000000.0))
    logger(f"seed: {seed}")

    if config.mode == 'train':
        start_train_epoch = 0


        scaler = amp.GradScaler()
        for current_epoch in range(start_train_epoch, config.total_train_epoch):


            model.lr_scheduler.step(current_epoch)

            _, result = train(model, loaders, config,scaler)
            logger('Time: {}; Epoch: {}; {}; {}'.format(time_now(), current_epoch, result,model.model_optimizer.state_dict()['param_groups'][0]['lr']))

            if (current_epoch + 1) % config.eval_epoch == 0:
                if config.dataset=='bupt':
                    cmc, mAP,eval_str = test_bupt(model, loaders, config)

                    if (cmc[1][0] + cmc[0][0]) >= best_cmc:

                        best_cmc = cmc[1][0] + cmc[0][0]
                        best_eval_str0 = eval_str[1]
                        best_eval_str1 = eval_str[0]
                        best_epoch = current_epoch

                    is_best_rank = (cmc[1][0] >= best_rank1)
                    best_rank1 = max(cmc[1][0], best_rank1)
                    model.save_model(current_epoch, is_best_rank)
                    logger('Time: {}; Test on Dataset: {}, \n task: {}, \n task: {}'.format(time_now(), config.dataset,eval_str[0],eval_str[1]))


                elif config.dataset=='vcm':
                    cmc, mAP,cmc2, mAP2, eval_str_t2v,eval_str_v2t = test_vcm(model, loaders)
                    if (cmc[0] + cmc2[0]) >= best_cmc:

                        best_cmc = cmc[0] + cmc2[0]
                        best_eval_str0 = eval_str_t2v
                        best_eval_str1 = eval_str_v2t
                        best_epoch = current_epoch

                    is_best_rank = (cmc[0] + cmc2[0] + (mAP + mAP2) * 100 >= best_rank1)
                    best_rank1 = max(cmc[0] + cmc2[0] + (mAP + mAP2) * 100, best_rank1)
                    model.save_model(current_epoch, is_best_rank)
                    logger('Time: {}; Test on Dataset: {}, \n task: {}, \n task: {}'.format(time_now(), config.dataset,eval_str_t2v,eval_str_v2t))

        if best_eval_str0 is not None:
            logger('Time: {}; Best Perform Test on Dataset: {}, , achieved at epoch {}, \n task: {}, \n task: {}'.format(time_now(), config.dataset, best_epoch,best_eval_str0,best_eval_str1))

    elif config.mode == 'test':
        model.resume_model(config.resume_test_path)
        if config.dataset == 'bupt':
            cmc, mAP, eval_str = test_bupt(model, loaders, config)
            logger('Time: {}; Test on Dataset: {}, \n task: {}, \n task: {}'.format(time_now(), config.dataset,
                                                                                    eval_str[0], eval_str[1]))
        elif config.dataset == 'vcm':
            cmc, mAP, cmc2, mAP2, eval_str_t2v, eval_str_v2t = test_vcm(model, loaders)
            logger('Time: {}; Test on Dataset: {}, \n task: {}, \n task: {}'.format(time_now(), config.dataset,
                                                                                    eval_str_t2v, eval_str_v2t))


    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger("Total running time: {}".format(total_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='train, test')
    parser.add_argument('--dataset', default='vcm', help='dataset name: vcm , bupt]')
    parser.add_argument('--vcm_data_path', type=str, default='/path/to/HITSZ-VCM')
    parser.add_argument('--bupt_data_path', type=str, default='/path/to/BUPTCampus')
    parser.add_argument('--batch-size', default=4, type=int, metavar='B', help='training batch size')
    parser.add_argument('--num_pos', default=4, type=int,
                        help='num of pos per identity in each modality')
    parser.add_argument('--img_w', default=144, type=int, metavar='imgw', help='img width')
    parser.add_argument('--img_h', default=288, type=int, metavar='imgh', help='img height')
    parser.add_argument('--test_batch', type=int, default=32)
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--pid_num', type=int, default=1074)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=0.000025)
    parser.add_argument('--lr_times', type=int, default=25)
    parser.add_argument('--num_workers', default=8, type=int,
                        help='num of pos per identity in each modality')

    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--output_path', type=str, default='logs/hha',
                        help='path to save related informations')
    parser.add_argument('--total_train_epoch', type=int, default=60)
    parser.add_argument('--eval_epoch', type=int, default=2)
    parser.add_argument('--resume_test_path', type=str, default='/path/to/checkpoint_directory')


    parser.add_argument('--train_frame_sample', type=str, default='random')
    parser.add_argument('--sequence_length', type=int, default=6)
    parser.add_argument('--random_flip', action='store_false', default=True)
    parser.add_argument('--fake', action='store_true', default=False)
    parser.add_argument('--test_frame_sample', type=str, default='uniform')
    parser.add_argument('--test_sampler', type=str,
                             default='ConsistentModalitySampler', help='None for no shuffle')
    parser.add_argument('--test_bs', type=int, default=64)
    parser.add_argument('--train_sampler', type=str,
                             default='RandomIdentitySampler', help='None for shuffle')
    parser.add_argument('--train_bs', type=int, default=16)
    parser.add_argument('--train_sampler_nc', type=int, default=2)
    parser.add_argument('--train_sampler_nt', type=int, default=1)
    parser.add_argument('--max_rank', type=int, default=20)


    config = parser.parse_args()

    main(config)
