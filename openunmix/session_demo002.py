from pathlib import Path
import torchaudio
import predict
import sessions
import torch
import json


if __name__ == '__main__':
    # Path
    mixture_path = './500greatest/002 - The Rolling Stones - Satisfaction (1965).flac'
    start = 5.0
    duration = 30.0
    # define model_targets
    model_targets = ['vocals','drums', 'bass']
    # parsing path
    mixture_path = Path(mixture_path)
    mixture_name = mixture_path.stem
    out_path = Path(mixture_name + '_results')
    out_path.mkdir(parents=True, exist_ok=True) 
    # load audio
    mixture, rate = torchaudio.load(mixture_path)
    # keep part
    mixture = mixture[:,int(start*rate):int(start*rate)+int(duration*rate)]

    torchaudio.save(
        str(Path(out_path,'session_' + mixture_name + '_ini.mp3')),
        torch.clamp(mixture, -1, 1),
        rate)

    # ini session
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
        [ ['drums'], 0.0 , 5.  ],
        [ ['bass'], 10. , 20.  ],
        [ ['drums', 'vocals'], 15. , 25.  ],
        [None, 15., 25.  ]
    ]

    # save scenario
    with open(Path(out_path, 'scenario.json'), 'w') as outfile:
            outfile.write(json.dumps({'scenario': test_values}, indent=4, sort_keys=True))
    
   
    for i, test in enumerate(test_values):
        targets, start, stop = test  
        if targets is None :
            config, audio = sessions.refine(
                config,
                audio,
                niter=1,
                use_residual=True,
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
