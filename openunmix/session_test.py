import argparse
from pathlib import Path
import torchaudio
import predict
import sessions
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='test of session object',
        add_help=False
    )
    parser.add_argument(
        '--mixture-path',
        type=str,
        default='/home/antoine/data/audio/1960s/Darling Baby - The Elgins-tSan37Algcg.wav',
        help='root path of dataset'
    ) 
    args, _ = parser.parse_known_args()
    # define model_targets
    model_targets = ['vocals','drums', 'bass']
    # parsing path
    mixture_path = Path(args.mixture_path)
    mixture_name = mixture_path.stem
    out_path = '.'
    # load audio
    mixture, rate = torchaudio.load(mixture_path)
    mixture = mixture[:, :rate*31]

    config, audio = sessions.init(
        mixture, # mixture signal ndarray
        rate, # sampling rate for the mixture
        model_targets, # list of str: targets handled by the model
        model_str_or_path="umxhq",
        model_rate=44100, # sampling rate for the model
        fade_len=0.025, # duration of fade-in/out
        device='cpu'
    )

    ##### tests : ORDER bass, drums, vocals
    # test of extract 
    test_values  = [
        [ ['bass', 'drums'], 0.0 , 5.  ],
        [ ['drums', 'bass'], 3.0 , 15.0  ],
        [ ['bass', 'drums', 'vocals'], 15 , 30  ],
        [None, 0.0, None  ]
    ]
    
   
    for i, test in enumerate(test_values):
        targets, start, stop = test  
        if targets is None :
            config, audio = sessions.refine(
                config,
                audio,
                niter=1,
                use_residual=False,
                start=start, stop=stop,
            )
        else : 
            # run extract method
            config, audio = sessions.extract(
                config=config, audio=audio,
                targets=targets,
                start=start,
                stop=stop)
        #  save 
        audio_torch = torch.as_tensor(audio)
        torchaudio.save(
            str(Path(out_path,'session_' + mixture_name + '_mix_' + str(i) + '.mp3')),
            torch.clamp(audio_torch[0], -1, 1),
            rate)
        for j, t in enumerate(model_targets):
            torchaudio.save(
                str(Path(out_path,'session_' + mixture_name + '_'  + t + '_'  + str(i) + '.mp3')),
                torch.clamp(audio_torch[j+1],-1,1),
                rate)
