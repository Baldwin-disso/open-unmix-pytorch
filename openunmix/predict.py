import argparse
import torchaudio
import torch
import model
from openunmix import utils


class Session:
    def __init__(
        self,
        mixture,
        rate,
        model_targets,
        model_str_or_path="umxhq",
        model_rate=44100.,
        fade_len=0.025,
        device='cpu',
    ):
        # save parameters
        self.model_str_or_path = model_str_or_path
        self.model_targets=model_targets
        self.model_rate=model_rate
        self.device=device

        # preparing audio of session
        mixture = utils.preprocess(mixture, rate, model_rate).to(device)
        self.audio = torch.zeros(
            (len(self.model_targets)+1, *mixture.shape[1:]),
            dtype=mixture.dtype,
            device=self.device
        )
        self.audio[0] = mixture

        # creating a fader
        self.fader = torchaudio.transforms.Fade(
            fade_in_len=int(fade_len * self.model_rate),
            fade_out_len=int(fade_len * self.model_rate),
            fade_shape='logarithmic'
        )

    def extract(
        self,
        targets,
        start=0.,
        stop=None
    ):
        separator = utils.load_separator(
            model_str_or_path=self.model_str_or_path,
            targets=targets,
            niter=0,
            residual=False,
            pretrained=True
        )
        separator.freeze()

        # get window to process
        start_sample = int(start * self.model_rate)
        stop_sample = -1 if stop is None else int(stop * self.model_rate)

        # getting the extracted signals
        extracted = separator(
            self.audio[0:1, :, start_sample:stop_sample]
        )
        for index, target in enumerate(targets):
            # add estimate to each target and remove from mix
            assert (
                target in self.model_targets,
                "Target %s must be in model_targets %s" % (
                    target, self.model_targets)
            )
            index_in_model = self.model_targets.index(target)
            faded_target = self.fader(extracted[0,index])

            # add to target waveform
            self.audio[1+index_in_model, :, start_sample:stop_sample] += faded_target
            # remove from mixture
            self.audio[0, :, start_sample:stop_sample] -= faded_target


    def refine(
            self,
            niter=1,
            use_residual=True,
            start=0, stop=None,
            n_fft = 4096,
            n_hop = 1024):

        # get excerpt to process
        start_sample = int(start * self.model_rate)
        stop_sample = -1 if stop is None else int(stop * self.model_rate)
        excerpt = self.fader(
            self.audio[..., start_sample:stop_sample]
        )
        # remove it from the current audio data
        self.audio[..., start_sample:stop_sample] -= excerpt

        # prepare the STFT object
        transform = model.STFT(n_fft=n_fft, n_hop=n_hop, center=True)
        abs = model.ComplexNorm(mono=self.audio.shape[1]==1)

        # apply it, get
        # (1+len(model_targets), nb_channels, nb_bins, nb_frames, complex=2)
        audio_stft = transform(excerpt)

        # go to 
        # (nb_frames, nb_bins, nb_channels, 1+len(model_targets), complex=2)
        audio_stft = torch.permute(audio_stft, [3, 2, 1, 0, 4])

        # (nb_frames, nb_bins, nb_channels, complex=2,
        #  len(models_targets) [+1 if residual])
        """result = wiener(
            abs(
                torch(audio_stft[..., 1:, :]
            ),
            audio_stft.sum(dim=-2),
            niter,
            softmask=False,
            residual=use_residual
        )"""

        # compute inverse STFT and store result
        self.audio[0, :, start_sample] += (
            model.istft(
                result[..., -1],
                n_fft=n_fft, n_hop=n_hop, center=True,
                length=self.audio.shape[-1]
            ) if use_residual
            else 0.
        )
        self.audio[1:] += model.istft(
            result[..., :len(self.model_targets)],
            n_fft=n_fft, n_hop=n_hop, center=True,
            length=self.audio.shape[-1]
        )


def inference_args(parser):
    inf_parser = argparse.ArgumentParser(
        description=__doc__,
        parents=[parser],
        add_help=True,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    inf_parser.add_argument(
        '--niter',
        type=int,
        default=1,
        help='number of iterations for refining results.'
    )

    inf_parser.add_argument(
        '--wiener-win-len',
        type=int,
        default=300,
        help='Number of frames on which to apply filtering independently'
    )

    inf_parser.add_argument(
        '--residual',
        type=str,
        default=None,
        help='if provided, build a source with given name'
             'for the mix minus all estimated targets'
    )

    inf_parser.add_argument(
        '--aggregate',
        type=str,
        default=None,
        help='if provided, must be a string containing a valid expression for '
             'a dictionary, with keys as output target names, and values '
             'a list of targets that are used to build it. For instance: '
             '\'{\"vocals\":[\"vocals\"], \"accompaniment\":[\"drums\",'
             '\"bass\",\"other\"]}\''
    )
    return inf_parser.parse_args()


def unmix(
    input_file,
    model_str_or_path="umxhq",
    targets=None,
    niter=1,
    residual=False,
    wiener_win_len=300,
    device='cpu',
    aggregate_dict=None,
    separator=None
):
    if separator is None:
        separator = utils.load_separator(
            model_str_or_path=model_str_or_path,
            targets=targets,
            niter=niter,
            residual=residual,
            wiener_win_len=wiener_win_len,
            device=device,
            pretrained=True
        )
        separator.freeze()

    # loop over the files
    # handling an input audio path
    audio, rate = torchaudio.load(input_file)
    audio = audio.to(device)
    audio = utils.preprocess(audio, rate, separator.sample_rate)

    # getting the separated signals
    estimates = separator(audio)
    estimates = separator.to_dict(
        estimates,
        aggregate_dict=aggregate_dict
    )
    return estimates
