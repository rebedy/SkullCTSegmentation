import os
import sys
from abc import ABC, abstractmethod
import argparse
import time
import numpy as np
import vtk
import torch

from args import parse_args
import utils
from utils import getMemoryUsage
from models import get_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)


class Strategy(ABC):

    @abstractmethod
    def do_algorithm(self):
        pass


class MoreSpeedStrategy(Strategy):

    def __init__(self, predNpArray, gtNpArray):

        self.gt1D = gtNpArray.ravel()
        self.pred1D = predNpArray.ravel()

    def do_algorithm(self):

        print("Start More Speed Strategy")

        start = time.time()

        gtTensor1D = torch.from_numpy(self.gt1D).to(device)
        predTensor1D = torch.from_numpy(self.pred1D).to(device)

        # predict mask
        pred_mask_0 = (predTensor1D == 0)
        pred_mask_1 = (predTensor1D == 1)
        pred_mask_2 = (predTensor1D == 2)

        del predTensor1D
        torch.cuda.empty_cache()

        # gt mask
        # gt : 0 , pred 0
        gt_mask_00 = (gtTensor1D == 0)
        result_00 = torch.logical_and(gt_mask_00, pred_mask_0)
        arr_00 = result_00.to(torch.uint8).cpu().numpy()
        del gt_mask_00
        del result_00
        torch.cuda.empty_cache()
        print("Running GPU Memory...: %s" % getMemoryUsage())

        # gt : 0 , pred 1
        gt_mask_10 = ((gtTensor1D + 1) == 1)
        result_10 = torch.logical_and(gt_mask_10, pred_mask_1)
        arr_10 = result_10.to(torch.uint8).cpu().numpy()
        del gt_mask_10
        del result_10
        torch.cuda.empty_cache()
        print("Running GPU Memory...: %s" % getMemoryUsage())

        # gt : 0 , pred 2
        gt_mask_20 = ((gtTensor1D + 2) == 2)
        result_20 = torch.logical_and(gt_mask_20, pred_mask_2)
        arr_20 = result_20.to(torch.uint8).cpu().numpy()
        del gt_mask_20
        del result_20
        torch.cuda.empty_cache()
        print("Running GPU Memory...: %s" % getMemoryUsage())

        # gt : 1, pred 0
        gt_mask_01 = ((gtTensor1D - 1) == 0)
        result_01 = torch.logical_and(gt_mask_01, pred_mask_0)
        arr_01 = result_01.to(torch.uint8).cpu().numpy()
        del gt_mask_01
        del result_01
        torch.cuda.empty_cache()
        print("Running GPU Memory...: %s" % getMemoryUsage())

        # gt : 1 , pred 1
        gt_mask_11 = (gtTensor1D == 1)
        result_11 = torch.logical_and(gt_mask_11, pred_mask_1)
        arr_11 = result_11.to(torch.uint8).cpu().numpy()
        del gt_mask_11
        del result_11
        torch.cuda.empty_cache()
        print("Running GPU Memory...: %s" % getMemoryUsage())

        # gt : 1, pred 2
        gt_mask_21 = ((gtTensor1D + 1) == 2)
        result_21 = torch.logical_and(gt_mask_21, pred_mask_2)
        arr_21 = result_21.to(torch.uint8).cpu().numpy()
        del gt_mask_21
        del result_21
        torch.cuda.empty_cache()
        print("Running GPU Memory...: %s" % getMemoryUsage())

        # gt : 2, pred 0
        gt_mask_02 = ((gtTensor1D - 2) == 0)
        result_02 = torch.logical_and(gt_mask_02, pred_mask_0)
        arr_02 = result_02.to(torch.uint8).cpu().numpy()
        del gt_mask_02
        del result_02
        torch.cuda.empty_cache()
        print("Running GPU Memory...: %s" % getMemoryUsage())

        # gt : 2, pred 1
        gt_mask_12 = ((gtTensor1D - 1) == 1)
        result_12 = torch.logical_and(gt_mask_12, pred_mask_1)
        arr_12 = result_12.to(torch.uint8).cpu().numpy()
        del gt_mask_12
        del result_12
        torch.cuda.empty_cache()
        print("Running GPU Memory...: %s" % getMemoryUsage())

        # gt : 2 , pred 2
        gt_mask_22 = (gtTensor1D == 2)
        result_22 = torch.logical_and(gt_mask_22, pred_mask_2)
        arr_22 = result_22.to(torch.uint8).cpu().numpy()
        del gt_mask_22
        del result_22
        torch.cuda.empty_cache()
        print("Running GPU Memory...: %s" % getMemoryUsage())

        del gtTensor1D
        del pred_mask_0
        del pred_mask_1
        del pred_mask_2
        torch.cuda.empty_cache()
        print("Running GPU Memory...: %s" % getMemoryUsage())

        # count
        sum_00 = np.sum(arr_00)
        sum_10 = np.sum(arr_10)
        sum_20 = np.sum(arr_20)
        sum_01 = np.sum(arr_01)
        sum_11 = np.sum(arr_11)
        sum_21 = np.sum(arr_21)
        sum_02 = np.sum(arr_02)
        sum_12 = np.sum(arr_12)
        sum_22 = np.sum(arr_22)

        confusion_list = [sum_00, sum_01, sum_02, sum_10, sum_11, sum_12, sum_20, sum_21, sum_22]
        confusion_matrix = np.array(confusion_list).reshape(3,3)

        TP = np.diag(confusion_matrix)
        FP = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
        FN = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
        TN = confusion_matrix.sum() - (FP + FN + TP)

        print("Confusion Matrix Computing Time: ", time.time() - start)
        print("End More Speed Strategy")

        return TP, FP, FN, TN


class MoreSafeStrategy(Strategy):

    def __init__(self, predNpArray, gtNpArray):

        self.filter = np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]])
        self.gt1D = gtNpArray.ravel()
        self.pred1D = predNpArray.ravel()
        self.divide = np.int8(np.ceil(self.gt1D.shape[0] * 0.5))
        self.gtFront1D = self.gt1D[0:self.divide]
        self.gtRear1D = self.gt1D[self.divide:]
        self.predFront1D = self.pred1D[0:self.divide]
        self.predRear1D = self.pred1D[self.divide:]
        self.frontArr = np.zeros((3, 3), dtype=object)
        self.rearArr = np.zeros((3, 3), dtype=object)

    def create_bit_mask(self, NpArray1D, class1, class2):
        if class1 == 0:
            mask = torch.from_numpy(NpArray1D).to(device)
            mask_bit = (mask == class2)

            del mask
            torch.cuda.empty_cache()

            return mask_bit

        else:
            mask = torch.from_numpy(NpArray1D).to(device)
            new_mask = mask + class1

            del mask
            torch.cuda.empty_cache()

            mask_bit = (new_mask == class2)

            del new_mask
            torch.cuda.empty_cache()

            return mask_bit

    def do_algorithm(self):

        print("Start More Safe Strategy")

        start = time.time()

        for i in range(3):
            for j in range(3):

                # front
                gt_mask = self.create_bit_mask(self.gtFront1D, self.filter[i][j], i)
                pred_mask = self.create_bit_mask(self.predFront1D, 0, i)
                result = torch.logical_and(gt_mask, pred_mask)
                self.frontArr[i][j] = np.sum(result.to(torch.uint8).cpu().numpy())
                print("Running GPU Memory...: %s" % getMemoryUsage())

                del gt_mask
                del pred_mask
                del result
                torch.cuda.empty_cache()
                print("Running GPU Memory...: %s" % getMemoryUsage())

                # rear
                gt_mask = self.create_bit_mask(self.gtRear1D, self.filter[i][j], i)
                pred_mask = self.create_bit_mask(self.predRear1D, 0, i)
                result = torch.logical_and(gt_mask, pred_mask)
                self.rearArr[i][j] = np.sum(result.to(torch.uint8).cpu().numpy())
                print("Running GPU Memory...: %s" % getMemoryUsage())

                del gt_mask
                del pred_mask
                del result
                torch.cuda.empty_cache()
                print("Running GPU Memory...: %s" % getMemoryUsage())

        confusion_matrix = self.frontArr + self.rearArr

        TP = np.diag(confusion_matrix)
        FP = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
        FN = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
        TN = confusion_matrix.sum() - (FP + FN + TP)

        print("Confusion Matrix Computing Time: ", time.time() - start)
        print("End More Safe Strategy")

        return TP, FP, FN, TN


class ConfusionMatrix():

    def __init__(self, predNpArray, gtNpArray) -> None:

        self.TP = None
        self.FP = None
        self.FN = None
        self.TN = None

        if (predNpArray.size < (550 * 768 * 768)):
            self._strategy = MoreSpeedStrategy(predNpArray, gtNpArray)
        else:
            self._strategy = MoreSafeStrategy(predNpArray, gtNpArray)

    @property
    def strategy(self) -> Strategy:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: Strategy) -> None:
        self._strategy = strategy

    def Compute(self) -> None:
        TP, FP, FN, TN = self._strategy.do_algorithm()

        self.TP = TP
        self.FP = FP
        self.FN = FN
        self.TN = TN

    def ComputePixelAccuracy(self):
        pixelAccuracy = (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)
        return pixelAccuracy.mean()

    def ComputeIOU(self):
        iou = self.TP / (self.TP + self.FP + self.FN)
        return iou.mean()

    def ComputeDiceCoefficent(self):
        diceCoefficent = 2 * self.TP / ((self.TP + self.FP) + (self.TP + self.FN))
        return diceCoefficent.mean()

    def ComputeDiceLoss(self):
        diceCoefficent = self.ComputeDiceCoefficent()
        return (1 - diceCoefficent).mean()


if __name__ == "__main__":

    print("Start GPU Memory Check: %s" % utils.getMemoryUsage())

    args = parse_args()

    # Check all used path
    utils.CheckFilePath(args.inputPath)
    utils.CheckFilePath(args.ckpt)

    # Create the standard renderer, render window and interactor
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    iren.SetRenderWindow(renWin)
    ren.SetBackground(1, 1, 1)
    renWin.SetSize(600, 600)

    # Find device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    checkpoint = torch.load(args.ckpt)
    classes = len(checkpoint["args"].mask) + 1

    # Initialize UNet
    model = get_model(args.model, args)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.train()
    print("MobileUNet initialization Done")

    # Read vti -> vtkImageData
    imageData = utils.ReadVTI(args.inputPath)

    npArray = utils.ConvertVtkToNumpy(imageData)

    # Predict image
    outputVolume = np.zeros(npArray.shape)

    for i in range(2, npArray.shape[0] - 2):

        imgSlice = npArray[i - 2: i + 3]

        inputTensor = torch.from_numpy(imgSlice).unsqueeze(0).to(torch.float32).to(device)

        pred = model.forward(inputTensor)

        mask = torch.argmax(pred, 1)
        masknp = mask.cpu().numpy()
        outputVolume[i] = masknp

        del inputTensor
        del pred
        del mask
        torch.cuda.empty_cache()

    print(np.min(outputVolume), np.max(outputVolume))

    # Convert numpy to vtk
    imageArray = utils.numpy_support.numpy_to_vtk(num_array=outputVolume.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)

    predImage = vtk.vtkImageData()
    predImage.DeepCopy(imageData)
    predImage.GetPointData().SetScalars(imageArray)
    predImage.Modified()

    # Render volume
    volume = utils.MakeVolume(predImage)
    volume.GetMapper().SetBlendModeToMaximumIntensity()
    ren.AddVolume(volume)

    # Convert vtkImageData to numpy array
    predNpArray = utils.ConvertVtkToNumpy(predImage)
    gtNpArray = utils.GenerateMaskImage(imageData, checkpoint["args"].mask)

    print("===========================================================")
    print("GPU Memory Before Confusion Matrix: %s" % getMemoryUsage())

    confusionMatrix = ConfusionMatrix(predNpArray, gtNpArray)
    confusionMatrix.Compute()

    print("===========================================================")
    print(confusionMatrix.ComputePixelAccuracy())
    print(confusionMatrix.ComputeIOU())
    print(confusionMatrix.ComputeDiceCoefficent())
    print("===========================================================")

    renWin.Render()
    iren.Initialize()
    renWin.Render()
    iren.Start()
