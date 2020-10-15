from pathlib import Path
import torchaudio
import predict
import sessions
import torch
import json


if __name__ == '__main__':
    # Path
    # specki
    """
    original_mixture_path = Path((
        "/media/speckbull/data/"
            "Rolling Stone Magazine's 500 Greatest Songs Of All Time/"
                "001-100/"
                    "002 - The Rolling Stones - Satisfaction (1965).flac"))
    """
    # saybi
    original_mixture_path = Path((
    "/home/baldwin/work/data/songs/satisfaction.mp3"))


    start = 5.0
    duration = 30.0
    #import ipdb; ipdb.set_trace()
    # define model_targets
    model_targets = ['vocals','drums', 'bass']

    # define session path template
    session_path_template = 'session_' + original_mixture_path.stem + '_{}'

    # load audio and truncate
    mixture, rate = torchaudio.load(original_mixture_path)
    mixture = mixture[:,int(start*rate):int(start*rate)+int(duration*rate)]

    # counter for the session recording
    counter = 0
    #import ipdb; ipdb.set_trace()
    # init session
    sessions.init(
        session_path_template.format(counter), # session path
        mixture, # mixture signal ndarray
        rate, # sampling rate for the mixture
        model_targets, # list of str: targets handled by the model
        model_str_or_path="umxhq",
        model_rate=44100, # sampling rate for the model
        fade_len=0.025, # duration of fade-in/out
        device='cpu',
        audio_format='mp3'
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
    with open(Path('scenario.json'), 'w') as outfile:
            outfile.write(json.dumps({'scenario': test_values}, indent=4, sort_keys=True))
    

    for (targets, start, stop) in test_values:
        config, audio = sessions.load(session_path_template.format(counter))
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

        counter = counter + 1
        sessions.save(config, audio, session_path_template.format(counter))
