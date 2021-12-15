import time
from datetime import timedelta
import torch
import logging
import utils.config as Config
import utils.logger as Logger
import utils.metrics as Metrics
from torch.utils.tensorboard import SummaryWriter
import os

from model.model import DDPM

from data import create_dataloader
from data.util import save_wav

def mainloop(phase, args):

    opt = Config.parse(args)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(phase, opt['path']['log'], level=logging.INFO, screen=True)

    logger = logging.getLogger(phase)
    logger.info(Config.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Calculate time points T to use
    if opt['model']['encoder']['type'] == 'conv':
        N = opt['model']['encoder']['conv']['N']
        L = opt['model']['encoder']['conv']['L']
        stride = opt['model']['encoder']['conv']['stride']
        # to make K equals L, T needs to be
        T = (N - 1) * stride + L
    else:
        raise NotImplementedError
    # dataset
    if phase == 'train':
        train_loader = create_dataloader('train', opt['sample_rate'], T, opt['datasets']['train'], logger)
    elif phase == 'val':
        val_loader = create_dataloader('val', opt['sample_rate'], T, opt['datasets']['val'], logger)
    elif phase == 'test':
        test_loader = create_dataloader('test', opt['sample_rate'], T, opt['datasets']['test'], logger)
    else:
        raise NotImplementedError
    logger.info('Initial Dataloader Finished')

    # model
    model = DDPM(phase, opt, logger)
    logger.info('Model [{:s}] is created.'.format(model.__class__.__name__))
    logger.info('Initial Model Finished')

    # Train
    current_step = model.begin_step
    current_epoch = model.begin_epoch

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    model.set_new_noise_schedule(
        opt['model']['beta_schedule'][phase], schedule_phase=phase)

    if phase == 'train':
        start = time.time()
        start_step = current_step
        n_iter = opt['train']['n_iter']
        while current_step < n_iter:
            current_epoch += 1
            for _, train_data in enumerate(train_loader):
                current_step += 1
                if current_step > n_iter:
                    break

                model.train(train_data)
                # log
                if current_step % opt['train']['print_freq'] == 0:
                    lapsed = time.time() - start
                    time_left =lapsed * (((n_iter - start_step) / (current_step-start_step)) - 1)
                    time_left = timedelta(seconds=time_left)
                    logs = model.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}, lapsed:{:.1f} remaining:{}> '.format(
                        current_epoch, current_step, lapsed, time_left)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)

                # validation
                if current_step % opt['train']['val_freq'] == 0:
                    # recreate validation dataset each time to get random data
                    val_loader = create_dataloader('val', opt['sample_rate'], T, opt['datasets']['val'], logger)
                    avg_sisnr = 0.0
                    idx = 0
                    result_path = '{}/{}'.format(opt['path']['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)

                    model.set_new_noise_schedule(
                        opt['model']['beta_schedule']['val'], schedule_phase='val')
                    for _,  val_data in enumerate(val_loader):
                        idx += 1

                        model.eval(val_data, continous=False)
                        sounds = model.get_current_sounds()

                        # log sisnr
                        avg_sisnr += Metrics.calculate_sisnr(
                             sounds['SR'], sounds['Clean'])

                    avg_sisnr = avg_sisnr / idx

                    model.set_new_noise_schedule(
                        opt['model']['beta_schedule']['train'], schedule_phase='train')
                    # log
                    logger.info('# Validation # SISNR: {:.4e}'.format(avg_sisnr))

                    # tensorboard logger
                    tb_logger.add_scalar('sisnr', avg_sisnr, current_step)

                if current_step % opt['train']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states.')
                    model.save_network(current_epoch, current_step)
        # save model
        logger.info('End of training.')

    elif phase == 'val' or phase == 'test':

        if phase == 'test':
            batch_size = opt['datasets']['test']
            loader = val_loader
            logger.info('Begin Model Test.')
        elif phase == 'val':
            batch_size = opt['datasets']['val']
            loader = test_loader
            logger.info('Begin Model Evaluation.')

        assert(batch_size == 1)
        sisnr_vec = torch.zeros(len(loader))
        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        for idx,  data in enumerate(loader):

            model.feed_data(data)
            model.eval(continous=True)
            sounds = model.get_current_sounds()

            # save the sound during the process

            sr_snd = sounds['SR']
            sample_num = sr_snd.shape[0]
            for iter in range(0, sample_num):
                save_wav(sr_snd[iter], '{}/{}_sr_{}.wav'.format(result_path, idx, iter))

            save_wav(
                sounds['Clean'], '{}/{}_clean.wav'.format(result_path,idx))
            save_wav(
                sounds['Noisy'], '{}/{}_noisy.wav'.format(result_path, idx))

            # metrics
            sisnr_vec[idx] = Metrics.calculate_sisnr(sr_snd[-1], sounds['Clean'])

        avg_sisnr = torch.mean(sisnr_vec)
        # log
        logger.info('# evaluation # SISNR: {:.4e}'.format(avg_sisnr))
        torch.save(sisnr_vec, '{}/sisnr_vec.pt'.format(result_path))

