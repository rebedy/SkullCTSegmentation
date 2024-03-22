
"""
Hausdorff loss implementation based on paper:
https://arxiv.org/pdf/1904.10030.pdf
copy pasted from - all credit goes to original authors:
https://github.com/SilmarilBearer/HausdorffLoss
"""
import os
import sys
import cv2
import numpy as np
import argparse
import torch
from torch import nn
import torch.optim as optim
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from scipy.ndimage.morphology import distance_transform_edt as edt
from scipy.ndimage import convolve

from args import parse_args
import utils
from models import get_model

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)


class HausdorffDTLoss(nn.Module):
    """Binary Hausdorff loss based on distance transform"""

    def __init__(self, alpha=2.0, **kwargs):
        super(HausdorffDTLoss, self).__init__()
        self.alpha = alpha

    @torch.no_grad()
    def test(self, img: np.ndarray) -> np.ndarray:
        return img

    @torch.no_grad()
    def distance_field(self, img: np.ndarray) -> np.ndarray:
        field = np.zeros_like(img)

        for batch in range(len(img)):
            fg_mask = img[batch] > 0.5

            if fg_mask.any():
                bg_mask = ~fg_mask

                fg_dist = edt(fg_mask)
                bg_dist = edt(bg_mask)

                field[batch] = fg_dist + bg_dist

        return field

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, debug=False
    ) -> torch.Tensor:
        """
        Uses one binary channel: 1 - fg, 0 - bg
        pred: (b, 1, x, y, z) or (b, 1, x, y)
        target: (b, 1, x, y, z) or (b, 1, x, y)
        """
        assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
        assert (
            pred.dim() == target.dim()
        ), "Prediction and target need to be of same dimension"

        # pred = torch.sigmoid(pred)
        # pred_test = torch.from_numpy(self.test( pred.detach().cpu().numpy() )).float()
        # print(pred_test) # no grad

        pred_dt = torch.from_numpy(self.distance_field(pred.detach().cpu().numpy())).float()
        target_dt = torch.from_numpy(self.distance_field(target.cpu().numpy())).float()

        pred_error = (pred - target) ** 2
        distance = pred_dt ** self.alpha + target_dt ** self.alpha

        dt_field = pred_error * distance
        loss = dt_field.mean()

        if debug:
            return (loss.cpu().numpy(), (dt_field.cpu().numpy()[0, 0],
                                         pred_error.cpu().numpy()[0, 0],
                                         distance.cpu().numpy()[0, 0],
                                         pred_dt.cpu().numpy()[0, 0],
                                         target_dt.cpu().numpy()[0, 0],),
                    )

        else:
            return loss


class HausdorffERLoss(nn.Module):
    """Binary Hausdorff loss based on morphological erosion"""

    def __init__(self, alpha=2.0, erosions=10, **kwargs):
        super(HausdorffERLoss, self).__init__()
        self.alpha = alpha
        self.erosions = erosions
        self.prepare_kernels()

    def prepare_kernels(self):
        cross = np.array([cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))])
        bound = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]])

        self.kernel2D = cross * 0.2
        self.kernel3D = np.array([bound, cross, bound]) * (1 / 7)

    @torch.no_grad()
    def perform_erosion(
        self, pred: np.ndarray, target: np.ndarray, debug
    ) -> np.ndarray:
        bound = (pred - target) ** 2

        if bound.ndim == 5:
            kernel = self.kernel3D
        elif bound.ndim == 4:
            kernel = self.kernel2D
        else:
            raise ValueError(f"Dimension {bound.ndim} is nor supported.")

        eroted = np.zeros_like(bound)
        erosions = []

        for batch in range(len(bound)):

            # debug
            erosions.append(np.copy(bound[batch][0]))

            for k in range(self.erosions):

                # compute convolution with kernel
                dilation = convolve(bound[batch], kernel, mode="constant", cval=0.0)

                # apply soft thresholding at 0.5 and normalize
                erosion = dilation - 0.5
                erosion[erosion < 0] = 0

                if erosion.ptp() != 0:
                    erosion = (erosion - erosion.min()) / erosion.ptp()

                # save erosion and add to loss
                bound[batch] = erosion
                eroted[batch] += erosion * (k + 1) ** self.alpha

                if debug:
                    erosions.append(np.copy(erosion[0]))

        # image visualization in debug mode
        if debug:
            return eroted, erosions
        else:
            return eroted

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, debug=False
    ) -> torch.Tensor:
        """
        Uses one binary channel: 1 - fg, 0 - bg
        pred: (b, 1, x, y, z) or (b, 1, x, y)
        target: (b, 1, x, y, z) or (b, 1, x, y)
        """
        assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
        assert (
            pred.dim() == target.dim()
        ), "Prediction and target need to be of same dimension"

        # pred = torch.sigmoid(pred)

        if debug:
            eroted, erosions = self.perform_erosion(
                pred.cpu().numpy(), target.cpu().numpy(), debug
            )
            return eroted.mean(), erosions

        else:
            eroted = torch.from_numpy(
                self.perform_erosion(pred.detach().cpu().numpy(), target.cpu().numpy(), debug)
            ).float()

            loss = eroted.mean()

            return loss


def readVTI(path):

    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(path)
    reader.Update()

    imageData = reader.GetOutput()

    return imageData


def GetArray(imageData):

    dims = imageData.GetDimensions()

    imageArray = vtk_to_numpy(imageData.GetPointData().GetScalars())
    imageArray = imageArray.reshape(dims[2], dims[0], dims[1])

    return imageArray


if __name__ == "__main__":

    args = parse_args()

    net = get_model(args.model, args)
    net.to("cpu")

    criterion = HausdorffDTLoss()
    opti = optim.Adam(net.parameters(), lr=0.001)

    parser = argparse.ArgumentParser(description='Hausdorff Loss Tester')
    parser.add_argument('--testData', type=str, default="//192.168.0.xxx/Data/000x.vti")
    args = parser.parse_args()

    # Read Sample Data and Mask
    imageData = utils.ReadVTI(args.testData)
    imageArray = utils.ConvertVtkToNumpy(imageData)
    maskArray = utils.GenerateMaskImage(imageData, ["mn"])

    # Get Sample Mask Slice
    mask = maskArray[120]
    image = imageArray[120]

    # Make Tensor
    input_tensor = torch.tensor([image]).unsqueeze(1).float()
    gt_tensor = torch.tensor([mask]).unsqueeze(1).float()
    # cv2.imwrite("temp/mask.jpg", mask*255)

    idx = 0
    while True:
        pred = net(input_tensor)
        pred = torch.nn.functional.softmax(pred, dim=1)

        pred_mask = pred[0][1].unsqueeze(0).unsqueeze(1)

        opti.zero_grad()
        loss = criterion(pred_mask, gt_tensor)
        loss.backward()
        opti.step()

        print("HDDT Loss : ", loss)

        pred_image = torch.argmax(pred, 1).to(torch.uint8)[0].numpy()
        cv2.imshow("mask", mask * 255)
        cv2.imshow("output", pred_image * 255)

        cv2.waitKey(1)
        # cv2.imwrite("temp/crossentropy_"+str(idx)+".jpg", pred_image*255)

        idx += 1

    cv2.waitKey(0)
    cv2.destroyAllWindows()
