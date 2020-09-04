import argparse
from pathlib import Path
import torchaudio
import predict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='test of session object',
        add_help=False
    )
    parser.add_argument(
        '--mixture-path',
        type=str,
        default='/home/baldo/work/source_separation/debug_data/session_test/R.E.M. - Losing My Religion.wav',
        help='root path of dataset'
    ) 
    args, _ = parser.parse_known_args()
    # define model_targets
    model_targets = ['vocals','drums', 'bass']
    # parsing path
    mixture_path = Path(args.mixture_path)
    mixture_name = mixture_path.stem
    out_path = mixture_path.parent
    # load audio
    mixture, rate = torchaudio.load(mixture_path)
    

    # create session object
    session = predict.Session(
        mixture,
        rate,
        model_targets,
        model_str_or_path="umxhq",
        model_rate=44100.,
        fade_len=0.025,
        device='cpu',
    )

    ##### tests : ORDER bass, drums, vocals
    # test of extract 
    test_values  = [
        [ ['drums'], 0.0 , 5.0  ],
        [ ['vocals'], 0.0 , 5.0  ],
        [ ['bass'], 0.0 , 5.0  ],
        [ ['drums','vocals'], 8.0 , 10.0  ],
        [ ['bass', 'drums', 'vocals'], 10.0 , 14.0  ],
    ]
    
   
    for i, test in enumerate(test_values):
        if i==3:
            print('break point')
        targets = test[0]; start = test[1]; stop = test[2]
        # run extract method
        session.extract(
            targets=targets,
            start=start,
            stop=stop)
        session.refine(
            niter=1,
            use_residual=True,
            start=start, stop=stop,
        )
        #  save 
        torchaudio.save(str(Path(out_path,'session_' + mixture_name + '_mix_' + str(i) + '.mp3')), session.audio[0], rate)
        for j, t in enumerate(model_targets):
            torchaudio.save(str(Path(out_path,'session_' + mixture_name + '_'  + t + '_'  + str(i) + '.mp3')), session.audio[j+1], rate)
