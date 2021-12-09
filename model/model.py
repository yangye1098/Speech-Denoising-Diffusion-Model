from collections import OrderedDict
import os

import torch
import torch.nn as nn
from .encoder_decoder import ConvDecoder, ConvEncoder
from .util import init_weights

####################
# define network
####################
class SpeechDenoisingUNet(nn.Module):
    def __init__(self, unet, encoder, decoder):
        super().__init__()
        self.unet = unet
        self.encoder = encoder
        self.decoder = decoder


    def forward(self, x_noisy, time, condition_x=None):
        time = torch.unsqueeze(time, 1)
        snd_rep = self.encoder(x_noisy, condition_x)
        denoised_rep = self.unet(snd_rep, time)
        return self.decoder(denoised_rep)


class DDPM():
    def __init__(self, phase, opt, logger):
        # initilize
        self.opt = opt
        self.device = torch.device(
            'cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.begin_step = 0
        self.begin_epoch = 0

        self.phase = phase
        self.logger = logger

        # initialize network and load pretrained models
        self.netG = self.init_model()

        self.schedule_phase = None

        # set loss and load resume state
        self.set_loss()
        self.set_new_noise_schedule(
           opt['model']['beta_schedule']['train'], schedule_phase='train')

        if phase == 'train':
            # find the parameters to optimize
            if opt['model']['finetune_norm']:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        self.logger.info(
                            'Params [{:s}] initialized to 0 and will optimize.'.format(k))
            else:
                optim_params = list(self.netG.parameters())

            if self.opt['train']['optimizer']['type'] == 'adam':
                self.optG = torch.optim.Adam(
                    optim_params, lr=opt['train']["optimizer"]["lr"])
            else:
                raise NotImplementedError
            self.log_dict = OrderedDict()

        self.load_network()
        self.print_network()

    def init_model(self):
        model_opt = self.opt['model']

        if model_opt['which_model_G'] == 'ddpm':
            raise NotImplementedError
        #    from .ddpm_modules import diffusion, unet
        elif model_opt['which_model_G'] == 'sr3':
            from .sr3_modules import diffusion, unet
        if ('norm_groups' not in model_opt['unet']) or model_opt['unet']['norm_groups'] is None:
            model_opt['unet']['norm_groups'] = 32

        if model_opt['encoder']['type'] == 'conv':
            encoder = ConvEncoder(
                N=model_opt['encoder']['conv']['N'],
                L=model_opt['encoder']['conv']['L'],
                stride=model_opt['encoder']['conv']['stride'],
            )
            decoder = ConvDecoder(
                N=model_opt['encoder']['conv']['N'],
                L=model_opt['encoder']['conv']['L'],
                stride=model_opt['encoder']['conv']['stride'],
            )
            image_size = model_opt['encoder']['conv']['N']
        else:
            raise NotImplementedError

        unet = unet.UNet(
            in_channel=model_opt['unet']['in_channel'],
            out_channel=model_opt['unet']['out_channel'],
            norm_groups=model_opt['unet']['norm_groups'],
            inner_channel=model_opt['unet']['inner_channel'],
            channel_mults=model_opt['unet']['channel_multiplier'],
            attn_res=model_opt['unet']['attn_res'],
            res_blocks=model_opt['unet']['res_blocks'],
            dropout=model_opt['unet']['dropout'],
            image_size=image_size
        )


        # initial weights of the unet
        init_type = 'orthogonal'
        self.logger.info('Initialization method [{:s}]'.format(init_type))
        init_weights(unet, init_type=init_type)

        model = SpeechDenoisingUNet(unet, encoder, decoder)

        netG = diffusion.GaussianDiffusion(
            model,
            channels=model_opt['diffusion']['channels'],
            loss_type=model_opt['diffusion']['loss_type'],  # L1 or L2
            conditional=model_opt['diffusion']['conditional'],
            schedule_opt=model_opt['beta_schedule']['train'],
        )

        # distributed
        if self.opt['gpu_ids'] and self.opt['distributed']:
            assert torch.cuda.is_available()
            netG = nn.DataParallel(netG)

        return netG

    def train(self, data):
        self.data = self.set_device(data)

        self.netG.train()
        self.optG.zero_grad()
        l_pix = self.netG(self.data)
        # need to average in multi-gpu
        nElement = torch.numel(self.data['Clean'])
        l_pix = l_pix.sum()/nElement
        l_pix.backward()
        self.optG.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()


    def eval(self, data, continous=False):
        self.data = self.set_device(data)
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.super_resolution(
                    self.data['Noisy'], continous)
            else:
                self.SR = self.netG.super_resolution(
                    self.data['Noisy'], continous)

    def sample(self, shape, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.sample(shape, continous)
            else:
                self.SR = self.netG.sample(shape, continous)
        self.netG.train()

    def set_loss(self):
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(
                    schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)

    def get_current_log(self):
        return self.log_dict

    def get_current_sounds(self, sample=False):
        out_dict = OrderedDict()
        if sample:
            out_dict['SAM'] = self.SR.detach().float().cpu()
        else:
            out_dict['SR'] = self.SR.detach().float().cpu()
            out_dict['Noisy'] = self.data['Noisy'].detach().float().cpu()
            out_dict['Clean'] = self.data['Clean'].detach().float().cpu()

        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        self.logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        self.logger.info(s)

    def save_network(self, epoch, iter_step):
        gen_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_gen.pth'.format(iter_step, epoch))
        opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
        # gen
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step,
                     'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.optG.state_dict()
        torch.save(opt_state, opt_path)

        self.logger.info(
            'Saved model in [{:s}] ...'.format(gen_path))

    def load_network(self):
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            self.logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)
            # gen
            network = self.netG
            if isinstance(self.netG, nn.DataParallel):
                network = network.module
            network.load_state_dict(torch.load(
                gen_path), strict=(not self.opt['model']['finetune_norm']))
            # network.load_state_dict(torch.load(
            #     gen_path), strict=False)
            if self.opt['phase'] == 'train':
                # optimizer
                opt = torch.load(opt_path)
                self.optG.load_state_dict(opt['optimizer'])
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']

    def set_device(self, x):
        if isinstance(x, dict):
            for key, item in x.items():
                if item is not None:
                    x[key] = item.to(self.device)
        elif isinstance(x, list):
            for item in x:
                if item is not None:
                    item = item.to(self.device)
        else:
            x = x.to(self.device)
        return x

    def get_network_description(self, network):
        '''Get the string and total parameters of the network'''
        if isinstance(network, nn.DataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n
