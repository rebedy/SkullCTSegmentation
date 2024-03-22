import vtk
from vtk.util import numpy_support
import numpy as np
import time
from subprocess import check_output
import os


def ReadVTI(vtkPath):
    # Read vti
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(vtkPath)
    reader.Update()

    # Get imageData
    return reader.GetOutput()


def MakeVolume(imageData):

    # Create transfer mapping scalar value to opacity
    opacityTransferFunction = vtk.vtkPiecewiseFunction()
    opacityTransferFunction.AddPoint(0, 0)
    opacityTransferFunction.AddPoint(1, 1)
    opacityTransferFunction.AddPoint(2, 1)
    opacityTransferFunction.AddPoint(3, 1)
    opacityTransferFunction.AddPoint(4, 1)

    # Create transfer mapping scalar value to color
    colorTransferFunction = vtk.vtkColorTransferFunction()
    colorTransferFunction.AddRGBPoint(0, 0.0, 0.0, 0.0)
    colorTransferFunction.AddRGBPoint(1, 1.0, 0.0, 0.0)
    colorTransferFunction.AddRGBPoint(2, 0.0, 1.0, 0.0)
    colorTransferFunction.AddRGBPoint(3, 0.0, 0.0, 1.0)
    colorTransferFunction.AddRGBPoint(4, 0.0, 0.6, 0.4)
    colorTransferFunction.AddRGBPoint(5, 0.5, 1.0, 0.2)

    # The property describes how the data will look
    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.SetColor(colorTransferFunction)
    volumeProperty.SetScalarOpacity(opacityTransferFunction)
    volumeProperty.ShadeOn()
    volumeProperty.SetInterpolationTypeToLinear()

    # The mapper / ray cast function know how to render the data
    volumeMapper = vtk.vtkSmartVolumeMapper()
    volumeMapper.SetInputData(imageData)

    # The volume holds the mapper and the property and
    # can be used to position/orient the volume
    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)

    return volume


def ConvertVtkToNumpy(image):

    # Get point data
    pointData = image.GetPointData().GetScalars()

    # Get dimensions
    dims = image.GetDimensions()

    # Convert vtkImageData to Numpy
    npArray = numpy_support.vtk_to_numpy(pointData)
    npArray = npArray.reshape(dims[2], dims[1], dims[0])

    return npArray


def GenerateMaskImage(imageData, maskName=["mx", "mn"]):

    pointData = imageData.GetPointData()
    dims = imageData.GetDimensions()

    generatedMaskArray = np.zeros((dims[2], dims[1], dims[0]))
    for idx, mask in enumerate(maskName):
        maskData = pointData.GetArray(mask)
        if not maskData:
            print(mask, "Doesn't exist")
            continue

        # Convert vtkImageData to Numpy
        maskArray = numpy_support.vtk_to_numpy(maskData)
        maskArray = maskArray.reshape(dims[2], dims[1], dims[0])
        maskArray = maskArray * (idx + 1)
        generatedMaskArray += maskArray
    generatedMaskArray = np.clip(generatedMaskArray, 0, 2)

    return generatedMaskArray


if __name__ == "__main__":

    print("Test Code ")

    testInputPath = "//192.168.0.xxx/train/00xxx.vti"

    testInputData = ReadVTI(testInputPath)

    start = time.time()
    try:
        maskArray = GenerateMaskImage(testInputData, ["mx", "mn"])
    except Exception as e:
        print(e)
        exit()

    print("Mask Array Generation : ", time.time() - start, "s")

    # Create the standard renderer, render window and interactor
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    ren.SetBackground(1, 1, 1)
    renWin.SetSize(600, 600)

    # Convert numpy to vtk
    maskData = numpy_support.numpy_to_vtk(num_array=maskArray.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)

    maskImageData = vtk.vtkImageData()
    maskImageData.DeepCopy(testInputData)
    maskImageData.GetPointData().SetScalars(maskData)
    maskImageData.Modified()

    maskVolume = MakeVolume(maskImageData)

    ren.AddVolume(maskVolume)

    renWin.Render()
    iren.Initialize()
    iren.Start()


def get_free_memory():
    output = check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.free"])
    output = output.decode('utf-8')
    output = output.split("\n")
    output = output[1:-1]

    gpu_idx = 0
    memory = 0

    for idx, value in enumerate(output):

        current_memory = int(value.split("MiB")[0])
        if current_memory > memory:
            memory = current_memory
            gpu_idx = idx
    print("available gpu : ", gpu_idx, ", memory : ", memory)

    return gpu_idx


def getMemoryUsage():
    # usage = nvsmi.DeviceQuery("memory.used")["gpu"][0]["fb_memory_usage"]
    # return "%f %s" % (usage["used"] / 954, "GB")

    output = check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.free"])
    output = output.decode('utf-8')
    output = output.split("\n")
    output = output[1:-1]

    return output


def CheckFilePath(path):
    isValidPath = os.path.exists(path)
    if not isValidPath:
        print("[ERROR] " + "invalid path: ", path)
        exit()
