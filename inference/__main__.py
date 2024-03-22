import os
import sys
import numpy as np
import onnxruntime
import torch
import vtk

from args import parse_args
from models import get_model
import utils

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


if __name__ == "__main__":

    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    ren.SetBackground(1, 1, 1)
    renWin.SetSize(600, 600)

    # Find device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = parse_args()

    # Load model
    input_path = "//192.168.0.xxx/Data/0000x.vti"
    onnx_path = "//192.168.0.xxx/checkpoints_epxxx.onnx"

    session = onnxruntime.InferenceSession(onnx_path)

    # Read vti -> vtkImageData
    imageData = utils.ReadVTI(input_path)
    npArray = utils.ConvertVtkToNumpy(imageData)

    model = get_model(args.model, args).to(device)

    # Predict image
    outputVolume = np.zeros(npArray.shape)

    for i in range(2, npArray.shape[0] - 2):

        imgSlice = npArray[i - 2: i + 3]

        inputTensor = torch.from_numpy(imgSlice).unsqueeze(0).to(torch.float32).to(device)

        ort_inputs = {"input": inputTensor.cpu().numpy()}

        ort_outs = session.run(None, ort_inputs)[0]

        masknp = np.argmax(ort_outs, 1)

        outputVolume[i] = masknp
