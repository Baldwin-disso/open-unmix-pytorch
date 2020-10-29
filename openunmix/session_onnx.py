from pathlib import Path
import torchaudio
import predict
import torch
import utils
import numpy as np
import torch.nn as nn

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()






def create_and_test_onnx_separator(
    input_mixture_path, 
    target,
    model='umxhq', 
    start = 0.0,
    duration = 20.0,
    do_creation = True,
    do_test = True,
):
    
    # parameters
    model_str_or_path = model 
    if Path(model).expanduser().exists(): # if model is a path
        model_name = Path(model_str_or_path).expanduser().stem
    else:
        model_name = model

    device = 'cpu'
    model_rate = 44100
    input_mixture_path = Path(input_mixture_path)
    onnx_model_format = 'separator-{}-{}-{}sec.onnx'.format(model_name, target, int(duration))
    onnx_out_format = '{}-{}-{}-onnx.mp3'.format(input_mixture_path.stem,model_name,target)
    pytorch_out_format = '{}-{}-{}-torch.mp3'.format(input_mixture_path.stem,model_name,target)
    pytorch_original_out_format = '{}-{}-{}-torch-original.mp3'.format(input_mixture_path.stem,model_name,target)

    # LOAD TORCH MODEL AND COMPUTE ESTIMATE 
    # device
    # model
    torch_separator = utils.load_separator(
        model_str_or_path=model_str_or_path,
        targets=[target],
        niter=0,
        residual=False,
        pretrained=True,
        use_original_umx = False
    ).to(device)
    torch_separator.freeze()
    # batch_size
    batch_size = 1    # just a random number

    # load audio and truncate
    mixture, rate = torchaudio.load(input_mixture_path)
    mixture = mixture[:,int(start*rate):int(start*rate)+int(duration*rate)]
    x = utils.preprocess(mixture, rate, model_rate)

    # inference
    torch_out = torch_separator(x)
    #import pdb; pdb.set_trace()
    # export to onnx 
    if do_creation: 
        # EXPORT to ONNX
        torch.onnx.export(torch_separator,               # model being run
                        x,                         # model input (or a tuple for multiple inputs)
                        onnx_model_format,   # where to save the model (can be a file or file-like object)
                        export_params=True,        # store the trained parameter weights inside the model file
                        opset_version=12,          # the ONNX version to export the model to
                        do_constant_folding=True,  # whether to execute constant folding for optimization
                        input_names = ['input'],   # the model's input names
                        output_names = ['output'], # the model's output names
                        dynamic_axes={'input' : { 2 : 'nb_steps'},    # variable lenght axes
                                        'output' : { 3 : 'nb_steps'}})
    if do_test:
        # Run original UMX model 
        torch_separator_original = utils.load_separator(
            model_str_or_path=model_str_or_path,
            targets=[target],
            niter=0,
            residual=False,
            pretrained=True,
            use_original_umx = True
        ).to(device)
        torch_separator_original.freeze()
        torch_out_original = torch_separator_original(x)
        # STORE original PYTORCH INFERENCE RESULT
        torchaudio.save(pytorch_original_out_format, torch_out_original[0,0],rate)

        #import pdb; pdb.set_trace()
        # TEST ONNX MODEL
        import onnx
 
        onnx_model = onnx.load(onnx_model_format)
        onnx.checker.check_model(onnx_model)

        # RUN ONNX MODEL
        import onnxruntime

        ort_session = onnxruntime.InferenceSession(onnx_model_format)
    
        # compute ONNX Runtime output prediction
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
        ort_outs = ort_session.run(None, ort_inputs)
    

        # STORE ONNX
        out_onnx = ort_outs[0]
        torchaudio.save(onnx_out_format, torch.tensor(out_onnx[0,0]), 44100)

        # STORE PYTORCH INFERENCE RESULT
        out_normal = to_numpy(torch_out)
        torchaudio.save(pytorch_out_format, torch_out[0,0],rate)
        
        #import pdb; pdb.set_trace()
        # III Compare Pytorch and ONNX
        # compare ONNX Runtime and PyTorch results
        np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
        print("onnx out and torch out are equals ! ")
        np.testing.assert_allclose(to_numpy(torch_out), to_numpy(torch_out_original) , rtol=1e-03, atol=1e-05)
        print("torch out  and original torch out are equals ! ")
        print("Exported model for : model " + model_name + " target  " + target +   " has been tested with ONNXRuntime, and the result looks good!")
    


if __name__ == '__main__':
    # EXPORT AND TEST SEPARATOR FOR ONNX
    input_mixture_path = "/home/baldwin/work/data/songs/satisfaction.mp3"
    input_mixture_path2 = "/home/baldwin/work/data/songs/closer.mp3"
    target_list = ['vocals', 'bass','drums']
    #model = '/home/baldwin/work/data/umx_models/UMX-PRO'
    model = 'umxhq'
    # create separators for target list
    
    for target in target_list:
        # create model with song satisfaction
        create_and_test_onnx_separator(
            input_mixture_path, 
            target, 
            model=model,  
            start = 10.0, 
            duration = 5.0,
            do_creation=True,
            do_test=True
        )
    
    # test separators for target list with another input
    for target in target_list:
        # test model with song 
        create_and_test_onnx_separator(
            input_mixture_path2, 
            target, 
            model=model,  
            start = 60.0, 
            duration = 20.0,
            do_creation=False,
            do_test=True
        )
    
    
    

        
    