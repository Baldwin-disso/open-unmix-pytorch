from pathlib import Path
import torchaudio
import predict
import torch
import utils
import numpy as np

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


if __name__ == '__main__':

    # INI MODEL
    # device
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    # model
    torch_separator = utils.load_separator(
        model_str_or_path="umxhq",
        targets=['vocals'],
        niter=0,
        residual=False,
        pretrained=True
    ).to(device)
    # batch_size
    batch_size = 1    # just a random number



    # INPUT FOR MODEL

    # saybi
    
    original_mixture_path = Path((
    "/home/baldwin/work/data/songs/satisfaction.mp3"))
    start = 5.0
    duration = 30.0
    # load audio and truncate
    mixture, rate = torchaudio.load(original_mixture_path)
    x = mixture[None,:,int(start*rate):int(start*rate)+int(duration*rate)]
    #x = utils.preprocess(mixture, 44100, 44100)
    '''
    x = torch.randn(batch_size, 2, 44100*5, requires_grad=True)
    '''
    torch_out = torch_separator(x)
    
    # EXPORT
    torch.onnx.export(torch_separator,               # model being run
                    x,                         # model input (or a tuple for multiple inputs)
                    "umx-vocals.onnx",   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=12,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    dynamic_axes={'input' : { 2 : 'nb_steps'},    # variable lenght axes
                                    'output' : { 3 : 'nb_steps'}})

    # onnx model TEST
    import onnx

    onnx_model = onnx.load("umx-vocals.onnx")
    onnx.checker.check_model(onnx_model)

    # onnx runtime test
    import onnxruntime

    ort_session = onnxruntime.InferenceSession("umx-vocals.onnx")
  
    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)
  

    # store onnx and pytorch results
    out_onnx = ort_outs[0]
    torchaudio.save('satis-vocals-onnx.mp3', torch.tensor(out_onnx[0,0]), 44100)
    
    #torch_out = torch_out/(2*torch_out.max())
    out_normal = to_numpy(torch_out)
    #import matplotlib.pyplot as plt
    #plt.plot(torch_out.detach().numpy()[0,0,0])
    #plt.show()
    #import pdb; pdb.set_trace()
    #torchaudio.save('satis-vocals-normal.mp3', torch.tensor(out_normal[0,0]), 44100)
    torchaudio.save('satis-vocals-normal_ast_norm.wav', torch_out[0,0],44100)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)


    print("Exported model has been tested with ONNXRuntime, and the result looks good!")
   

        

   