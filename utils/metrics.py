import torch
# from torchmetrics.functional import pesq
# from torchmetrics.functional import stoi


def calculate_sisnr(s_hat, s):
    """
    Calculate SISNR
    Args:
        s_hat: [B, 1, T] or [B, T]
        s: [B, 1, T] or [B, T] the true sources
    Returns:
        SI-SNR: [1]

    """
    if s_hat.ndim == 2:
        s_hat = s_hat.unsqueeze(1)

    if s.ndim == 2:
        s = s.unsqueeze(1)

    print(s_hat.shape)
    print(s.shape)

    # normalize to zero mean
    s_hat = s_hat - torch.mean(s_hat, 2, keepdim=True)  # [B, 1, T]
    s = s - torch.mean(s, 2, keepdim=True)  # [B, 1, T]
    # <s, s_hat>s/||s||^2
    s_shat = torch.sum(s_hat * s, dim=2, keepdim=True)  # [B, 1, 1]
    s_2 = torch.sum(s ** 2, dim=2, keepdim=True)  # [B, 1, T]
    s_target = s_shat * s / s_2  # [B, 1, T]

    # e_noise = s_hat - s_target
    e_noise = s_hat - s_target  # [B, 1, T]
    sisnr = 10 * torch.log10(torch.sum(s_target ** 2, dim=2, keepdim=True) \
                            / torch.sum(e_noise ** 2, dim=2, keepdim=True)) # [B, 1, T]

    return torch.squeeze(torch.mean(sisnr))


def calculate_PESQ():
    pass

def calculate_CSIG():
    pass

def calculate_STOI():
    pass

def calculate_CBAK():
    pass

def calculate_COVL():
    pass



if __name__ == '__main__':
    from data import AudioDataset
    snr = 0

    dataroot = f'../data/data/wsj0_si_tr_{snr}'
    datatype = 'wav'
    dataset_tr = AudioDataset(dataroot, datatype, snr, 33224)
    dataset_tr.playIdx(0)
    s = dataset_tr.__getitem__(0)
    clean = torch.unsqueeze(s['Clean'], dim=0)
    noisy = torch.unsqueeze(s['Noisy'], dim=0)

    for i in range(1,4):
        s = dataset_tr.__getitem__(i)
        clean = torch.cat([clean, torch.unsqueeze(s['Clean'], 0)], dim=0)
        noisy = torch.cat([noisy, torch.unsqueeze(s['Noisy'], dim=0)], dim = 0)

    print(clean.shape)
    print(calculate_sisnr(clean, clean))
    print(calculate_sisnr(noisy, clean))
