import argparse
import torchaudio
import torch
from openunmix import model
from openunmix import filtering
from openunmix import utils
import numpy as np


def init(
    mixture, # mixture signal ndarray
    rate, # sampling rate for the mixture
    model_targets, # list of str: targets handled by the model
    model_str_or_path="umxhq",
    model_rate=44100, # sampling rate for the model
    fade_len=0.025, # duration of fade-in/out
    device='cpu'
):
    # preparing audio of session
    mixture = utils.preprocess(mixture, rate, model_rate)
    audio = np.zeros(
        (len(model_targets)+1, *mixture.shape[1:]),
    )
    audio[0] = mixture.numpy()

    config = {
        'model_str_or_path': model_str_or_path,
        'model_targets': model_targets,
        'model_rate': model_rate,
        'device': device,
        'fade_len': fade_len,
        'processed_targets': set()
    }
    return config, audio

def extract(
    config,
    audio,
    targets,
    start=0.,
    stop=None
):
    separator = utils.load_separator(
        model_str_or_path=config['model_str_or_path'],
        targets=targets,
        niter=0,
        residual=False,
        pretrained=True
    ).to(config['device'])
    separator.freeze()

    # send to torch and device
    audio = torch.as_tensor(audio, device=config['device'], dtype=torch.float32)

    # creating a fader
    fader = torchaudio.transforms.Fade(
        fade_in_len=int(config['fade_len'] * config['model_rate']),
        fade_out_len=int(config['fade_len'] * config['model_rate']),
        fade_shape='logarithmic'
    )

    # get window to process
    start_sample = int(start * config['model_rate'])
    stop_sample = audio.shape[-1] if stop is None else int(stop * config['model_rate'])

    # getting the extracted signals applied on the mixture
    extracted = separator(
        audio[None, 0, :, start_sample:stop_sample]
    )[0]
    #import matplotlib.pylab as plt
    for index, target in enumerate(separator.target_models.keys()):
        # add estimate to each target and remove from mix
        assert target in config['model_targets'],\
            "Target %s must be in model_targets %s" % (
                target, config['model_targets'])

        index_in_audio = config['model_targets'].index(target)
        faded_target = fader(extracted[index])

        # add to target waveform
        audio[1+index_in_audio, :, start_sample:stop_sample] += faded_target
        # remove from mixture
        audio[0, :, start_sample:stop_sample] -= faded_target
    
    config['processed_targets'] = config['processed_targets'].union(targets)

    return config, audio.cpu().numpy()


def refine(
        config,
        audio,
        niter=1,
        use_residual=True,
        start=0, stop=None,
        n_fft = 4096,
        n_hop = 1024):
    #import pdb; pdb.set_trace()
    # get excerpt to process
    start_sample = int(start * config['model_rate'])
    stop_sample = audio.shape[-1] if stop is None else int(stop * config['model_rate'])

    # creating a fader
    fader = torchaudio.transforms.Fade(
        fade_in_len=int(config['fade_len'] * config['model_rate']),
        fade_out_len=int(config['fade_len'] * config['model_rate']),
        fade_shape='logarithmic'
    )

    # don't select the targets that have not been processed
    selected_indexes = [0,] + [
        index+1 for (index, target) in enumerate(config['model_targets'])
        if target in config['processed_targets']]
    n_selected = len(selected_indexes)

    # get the excerpt for the selected targets + mixture
    excerpt = torch.as_tensor(
        audio[selected_indexes, :, start_sample:stop_sample],
        device=config['device']
    )

    # apply the fader
    excerpt = fader(excerpt)
    # keep current excerpt 
    old_excerpt = excerpt.cpu().numpy()

    # remove it from the current audio data
    audio[selected_indexes, :, start_sample:stop_sample] -= excerpt.cpu().numpy()


    # prepare the STFT object
    transform = model.STFT(n_fft=n_fft, n_hop=n_hop, center=True)

    # apply it, get
    # (n_selected, nb_channels, nb_bins, nb_frames, complex=2)
    audio_stft = transform(excerpt)

    # go to 
    # (nb_frames, nb_bins, nb_channels, complex=2, n_selected)
    audio_stft = audio_stft.permute(3, 2, 1, 4, 0)

    # compute the mixture (nb_frames, nb_bins, nb_channels, complex=2)
    mixture_stft = audio_stft.sum(dim=-1)

    if not use_residual:
        # trim the first element ('residual')
        audio_stft = audio_stft[...,1:]

    # get the refined signals (nb_frames, nb_bins, nb_channels, 2, n_selected)
    audio_stft, _, _ = filtering.expectation_maximization(
        y=audio_stft,
        x=mixture_stft,
        iterations=niter,        
    )

    # go to (n_selected, nb_channels, nb_bins, nb_frames, complex=2) (in stft)
    audio_stft = audio_stft.permute(4, 2, 1, 0, 3 )

    audio_wave = model.istft(
            audio_stft,
            n_fft=n_fft, n_hop=n_hop, center=True,
            length=stop_sample - start_sample
        ).cpu().numpy()

    # normalize audio_wave to fit excerpt
    f = (((old_excerpt**2).sum())/((audio_wave**2).sum()))**(1/2)
    audio_wave *= f 
    print('factor_used' + str(f))

    # update audio
    pos = 0
    if use_residual:
        # if we use residual: update the first element from audio
        audio[0,:,start_sample:stop_sample] += audio_wave[0]
        pos = 1
    audio[1:, :, start_sample:stop_sample] += audio_wave[pos:]

    return config, audio

