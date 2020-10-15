from pathlib import Path
import torchaudio
import predict
import torch
import utils
import numpy as np



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
    x = torch.randn(batch_size, 2, 44100*5, requires_grad=True)
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

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    out_normal = to_numpy(torch_out)
    out_onnx = ort_outs[0]
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")
    import pdb; pdb.set_trace()

        

   