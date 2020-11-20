import torchaudio
import predict
import torch
import utils
import numpy as np
import torch.nn as nn
from asteroid.dsp import LambdaOverlapAdd



class SessionExtractor(nn.Module):
    def __init__(
        self,
        model_str_or_path,
        extraction_target = 'vocals',
        session_targets=['drums', 'bass', 'vocals'],
        model_rate=44100, 
        niter=0,
        residual=False,
        pretrained=True,
        use_original_umx = False,
        device = 'cpu',
        fade_len = 0.025,
        window_size = 5.0
    ):
        super(SessionExtractor, self).__init__()
        # store parameters 
        self.model_str_or_path = model_str_or_path
        self.extraction_target = extraction_target
        self.session_targets = session_targets
        self.model_rate = model_rate
        self.niter = niter
        self.residual = residual
        self.pretrained = pretrained
        self.use_original_umx = use_original_umx
        self.device = device
        self.fade_len = fade_len
        self.window_size = 5.0
    

        # instanciate separator as overlap-add
        self.separator = utils.load_separator(
            model_str_or_path=self.model_str_or_path,
            targets=[self.extraction_target],
            niter=self.niter,
            residual=self.residual,
            pretrained=self.pretrained,
            use_original_umx = self.use_original_umx
        )
        self.separator.freeze()
        """
        self.extractor = LambdaOverlapAdd(
            self.separator,
            n_src=1,
            window_size = self.window_size,
            hop_size=None,
            window="hanning",
            reorder_chunks=True,
            enable_grad=False,
            )
        """
        # creating a fader
        self.fader = torchaudio.transforms.Fade(
            fade_in_len=int(self.fade_len * self.model_rate),
            fade_out_len=int(self.fade_len * self.model_rate),
            fade_shape='logarithmic'
        )
    def forward(self, session, start, stop):
        # session : (session, channels, steps)
        # get window to process
        session_out = session.clone()
        start_sample = int(start * self.model_rate)
        stop_sample = session_out.shape[-1] if stop is None else int(stop * self.model_rate)
        duration = stop - start
        # getting the extracted signals applied on the mixture
        extracted = self.separator(
            session_out[None, 0, :, start_sample:stop_sample]
        )[0]
        extracted = extracted[...,:int(duration*self.model_rate)]
        # get index of session to update
        index_in_audio = self.session_targets.index(self.extraction_target)
        faded_target = self.fader(extracted[0])
        # add to target waveform
        session_out[1+index_in_audio, :, start_sample:stop_sample] += faded_target
        # remove from mixture
        session_out[0, :, start_sample:stop_sample] -= faded_target
        return session_out


if __name__ == '__main__':

    # TEST OF EXTRACTION CLASS
    
    input_mixture_path = "/home/baldwin/work/data/songs/satisfaction.mp3"
    target_list = ['vocals', 'bass','drums']
    model = '/home/baldwin/work/data/umx_models/UMX-PRO'
    onnx_model_format  = 'session_extractor_vocals.onnx'

    # instanciate SessionExtrator

    session_extractor = SessionExtractor(
            model_str_or_path=model,
            extraction_target = 'vocals',
            session_targets=target_list,
            model_rate=44100, 
            niter=0,
            residual=False,
            pretrained=True,
            use_original_umx = False,
            device = 'cpu',
            fade_len = 0.025,
            window_size = 5.0
        )

    # create Session
    # load audio and truncate
    mixture, rate = torchaudio.load(input_mixture_path)
    x = utils.preprocess(mixture, rate, session_extractor.model_rate)
    # create zeros to ini session
    session = torch.zeros((1+len(target_list),) + x[0].shape)
    session[0] = x[0]
    start = torch.tensor(0.0)
    stop = torch.tensor(5.0)
    session_out = session_extractor(session, start, stop ) 
    
    do_creation = True
    if do_creation: 
        # EXPORT to ONNX
        torch.onnx.export(session_extractor,               # model being run
                        (session, torch.tensor(0.0), torch.tensor(5.0)),                         # model input (or a tuple for multiple inputs)
                        'session_extractor_vocals.onnx',   # where to save the model (can be a file or file-like object)
                        export_params=True,        # store the trained parameter weights inside the model file
                        opset_version=12,          # the ONNX version to export the model to
                        do_constant_folding=True,  # whether to execute constant folding for optimization
                        input_names = ['besh','start','stop'],   # the model's input names
                        output_names = ['output'], # the model's output names
        )
    
    do_test = True
    if do_test:
    
        # TEST ONNX MODEL
        import onnx
        
        onnx_model = onnx.load(onnx_model_format)
        onnx.checker.check_model(onnx_model)
        
        # RUN ONNX MODEL
        import onnxruntime
        
        ort_session = onnxruntime.InferenceSession(onnx_model_format)
        
        # compute ONNX Runtime output prediction
        ort_inputs = {
            'besh': onnx_model_format, 
            'start':  start,
            'stop': stop
        }
        
        import pdb; pdb.set_trace()
        ort_outs = ort_session.run(None, ort_inputs)
        
        # STORE PYTORCH INFERENCE RESULT
        out_normal = to_numpy(session_out)
        
        # III Compare Pytorch and ONNX
        # compare ONNX Runtime and PyTorch results
        np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
        print("Exported model for : model " + model_name + " target  " + target +   " has been tested with ONNXRuntime, and the result looks good!")
