import time
from datetime import timedelta
import torch
import logging
import utils.config as Config
import utils.logger as Logger
import utils.metrics as Metrics
from utils.gram_transform import ReverseRISpectrograms
from torch.utils.tensorboard import SummaryWriter
import os

from model.model import DDPM

from data import create_dataloader
from torchvision.utils import save_image
import torchaudio
import numpy as np

def mainloop(phase, args):

    opt = Config.parse(args)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(phase, opt['path']['log'], level=logging.INFO, screen=True)

    logger = logging.getLogger(phase)
    logger.info(Config.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    datatype = opt['datatype']
    sample_rate = opt['sample_rate']
    # Calculate time points T to use

    encoder_opt = opt['model']['encoder']
    encoder_type = encoder_opt['type']
    if encoder_type in ['RI_mel', 'RI']:
        N = encoder_opt[encoder_type]['N']
        L = encoder_opt[encoder_type]['L']
        stride = encoder_opt[encoder_type]['stride']
        expand_order = encoder_opt[encoder_type]['expand_order']
        T = (N-1)*stride
        use_mel = encoder_type == 'RI_mel'
        gram_decoder = ReverseRISpectrograms(N, L, stride, sample_rate, expand_order, use_mel)
    else:
        raise NotImplementedError


    # dataset
    if phase == 'train':
        train_loader = create_dataloader(sample_rate, datatype, T, opt['datasets']['train'], logger)
    elif phase == 'val':
        val_loader = create_dataloader(sample_rate, datatype, T, opt['datasets']['val'], logger)
    elif phase == 'test':
        test_loader = create_dataloader(sample_rate, datatype, T, opt['datasets']['test'], logger)
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

    if opt['path']['resume_state'] and phase == 'train':
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    model.set_new_noise_schedule(
        opt['model']['beta_schedule'][phase], schedule_phase=phase)

    if phase == 'train':
        start = time.time()
        start_step = current_step
        n_iter = opt['train']['n_iter']
        val_iter = opt['train']['val_iter']
        while current_step < n_iter:
            current_epoch += 1
            for _, (target, noisy, _) in enumerate(train_loader):
                target, noisy = target.to(model.device), noisy.to(model.device)
                current_step += 1
                if current_step > n_iter:
                    break

                model.train(target, noisy)
                # log
                if current_step % opt['train']['print_freq'] == 0:
                    lapsed = time.time() - start
                    time_left = lapsed * (((n_iter - start_step) / (current_step-start_step)) - 1)
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
                    val_loader = create_dataloader(sample_rate, datatype, T, opt['datasets']['val'], logger)
                    avg_sisnr = 0.0
                    idx = 0
                    result_path = '{}/{}'.format(opt['path']['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)

                    model.set_new_noise_schedule(
                        opt['model']['beta_schedule']['val'], schedule_phase='val')
                    for _,  (target, noisy, _) in enumerate(val_loader):
                        if idx >= val_iter:
                            break
                        target, noisy = target.to(model.device), noisy.to(model.device)
                        idx += 1

                        SR = model.eval(noisy, continuous=False)

                        # log sisnr

                        if datatype == '.wav':
                            avg_sisnr += Metrics.calculate_sisnr(
                                SR, target)
                        elif datatype == '.spec.npy' or datatype == '.mel.npy':
                            gram_decoder.to(model.device)
                            SR_sound = gram_decoder(SR)
                            target_sound = gram_decoder(target)
                            avg_sisnr += Metrics.calculate_sisnr(
                                SR_sound, target_sound)

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
            n_iter = opt['train']['val_iter']
            batch_size = opt['datasets']['test']['batch_size']
            loader = test_loader
            logger.info('Begin Model Test.')
        elif phase == 'val':
            n_iter = opt['train']['val_iter']
            batch_size = opt['datasets']['val']['batch_size']
            loader = val_loader
            logger.info('Begin Model Evaluation.')


        if n_iter < 0:
            metric_vec = torch.zeros(len(loader))
        else:
            metric_vec = torch.zeros(n_iter)


        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        result_path = '{}'.format(opt['path']['results'])
        os.makedirs('{}/clean'.format(result_path), exist_ok=True)
        os.makedirs('{}/noisy'.format(result_path), exist_ok=True)
        os.makedirs('{}/output'.format(result_path), exist_ok=True)
        for idx,  (clean, noisy, name_index) in enumerate(loader):
            if n_iter >= 0 and idx > n_iter:
                break
            clean, noisy = clean.to(model.device), noisy.to(model.device)
            output = model.eval(noisy, continuous=False)

            #for iter in range(0, sample_num):

            if datatype == '.wav':
                for b in range(batch_size):
                    name = loader.dataset.getName(name_index[b])
                    torchaudio.save('{}/output/{}.wav'.format(result_path, name), output[b, :, :], sample_rate)
                    torchaudio.save('{}/clean/{}.wav'.format(result_path, name), clean[b, :, :], sample_rate)
                    torchaudio.save('{}/noisy/{}.wav'.format(result_path, name), noisy[b, :, :], sample_rate)


                metric_vec[idx] = Metrics.calculate_sisnr(output, clean)
            elif datatype == '.spec.npy':

                gram_decoder.to(model.device)
                output_sound = gram_decoder(output)
                clean_sound = gram_decoder(clean)
                noisy_sound = gram_decoder(noisy)
                metric_vec[idx] = Metrics.calculate_sisnr(output_sound, clean_sound)

                for b in range(batch_size):
                    name = loader.dataset.getName(name_index[b])
                    np.save('{}/output/{}.spec.npy'.format(result_path, name), output[b,:, :, :].cpu().numpy())
                    torchaudio.save('{}/output/{}.spec.npy.wav'.format(result_path, name), output_sound[b, :].cpu(), sample_rate)
                    np.save(
                        '{}/clean/{}.spec.npy'.format(result_path, name), clean[b,:, :, :].cpu().numpy())
                    torchaudio.save('{}/clean/{}.spec.npy.wav'.format(result_path, name), clean_sound[b, :].cpu(), sample_rate)
                    np.save(
                        '{}/noisy/{}.spec.npy'.format(result_path, name), noisy[b,:, :, :].cpu().numpy())
                    torchaudio.save('{}/noisy/{}.spec.npy.wav'.format(result_path, name), noisy_sound[b, :].cpu(), sample_rate)


            elif datatype == '.mel.npy':
                for b in range(batch_size):
                    name = loader.dataset.getName(name_index[b])
                    save_image(torch.flip(output[b, :, :, :], [1]), '{}/output/{}.png'.format(result_path, name))
                    save_image(
                        torch.flip(clean[b, :, :, :], [1]), '{}/clean/{}.png'.format(result_path, name))
                    save_image(
                        torch.flip(noisy[b, :, :, :], [1]), '{}/noisy/{}.png'.format(result_path, name))

        if datatype == '.wav':
        # metrics
            avg_metric = torch.mean(metric_vec)
            # log
            logger.info('# evaluation # SISNR: {:.4e}'.format(avg_metric))
            torch.save(metric_vec.cpu(), '{}/sisnr_vec.pt'.format(result_path))
        elif datatype == '.spec.npy':
            avg_metric = torch.mean(metric_vec)
            # log
            logger.info('# evaluation # SISNR: {:.4e}'.format(avg_metric))
            torch.save(metric_vec.cpu(), '{}/sisnr_vec.pt'.format(result_path))
        elif datatype == '.mel.npy':
            pass


