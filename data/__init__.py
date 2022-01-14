import torch
SND_DTYPE = 'float32'

from .dataset import AudioDataset

def create_dataloader(sample_rate, datatype, T, dataset_opt, logger):
    '''create dataloader '''
    dataset = AudioDataset(dataroot=dataset_opt['dataroot'],
                           datatype=datatype,
                           sample_rate=sample_rate,
                           T = T
                           )

    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            shuffle=dataset_opt['use_shuffle'],
            num_workers=dataset_opt['num_workers'],
            pin_memory=True)





