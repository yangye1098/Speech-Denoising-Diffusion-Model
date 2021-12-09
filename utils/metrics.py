import torch


def calculate_sisnr(s_hat, s):
    """
    Calculate SISNR
    Args:
        s_hat: [B, 1, T]
        s: [B, 1, T] the true sources
    Returns:
        SI-SNR: [1]

    """
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
