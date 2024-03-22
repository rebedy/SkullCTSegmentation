import os
import sys
import argparse
import glob

from PyQt5.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QListWidget, QApplication, QComboBox
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtk

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)


class Window(QMainWindow):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)        

        # Set up GUI
        self.setWindowTitle("ImagoWorksCTDataChecker")
        self.setCentralWidget(QWidget())
        self.centralWidget().setLayout(QHBoxLayout())

        # list widget
        self.listWidget = QListWidget()
        self.comboBox = QComboBox()
        self.comboBox.textActivated.connect(self.ComboBoxChanged)

        # Initialize VTK
        self.iren = QVTKRenderWindowInteractor()
        self.iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        self.ren = vtk.vtkRenderer()
        self.ren.SetBackground(0, 0, 0)
        self.renWin = self.iren.GetRenderWindow()
        self.renWin.AddRenderer(self.ren)

        # Initialize Widgets
        self.centralWidget().layout().addWidget(self.listWidget, 10)
        self.centralWidget().layout().addWidget(self.iren, 80)
        self.centralWidget().layout().addWidget(self.comboBox)

        self.renWin.Render()

        # Initialize Rendering Pipeline
        self.InitVTKPipeline()

    def InitVTKPipeline(self):

        # Initialize OTF, CTF
        self.opacityTransferFunction = vtk.vtkPiecewiseFunction()        
        self.colorTransferFunction = vtk.vtkColorTransferFunction()

        self.opacityTransferFunction.AddPoint(-1024, 0.0)
        self.opacityTransferFunction.AddPoint(3000 , 0.5)
        self.colorTransferFunction.AddRGBPoint(0, 0, 0, 0)
        self.colorTransferFunction.AddRGBPoint(1, 1, 1, 1)
        # self.colorTransferFunction.AddRGBPoint(2, 0, 1, 0)
        # self.colorTransferFunction.AddRGBPoint(3, 0, 0, 1)

        self.imageData = None
        self.mapper = vtk.vtkSmartVolumeMapper()

        self.volume = vtk.vtkVolume()
        self.volume.SetMapper(self.mapper)
        self.volume.GetProperty().SetScalarOpacity(self.opacityTransferFunction)
        self.volume.GetProperty().SetColor(self.colorTransferFunction)
        self.volume.GetProperty().SetInterpolationTypeToLinear()
        self.volume.GetProperty().ShadeOn()

        self.ren.AddViewProp(self.volume)
        self.ren.ResetCamera()
        self.renWin.Render()

    def SetDataDir(self, dataDir):
        print("Loading Mask List.....")
        maskList = glob.glob(os.path.join(dataDir, "**/*.vti"), recursive=True)
        self.listWidget.itemDoubleClicked.connect(self.ItemClicked)
        for maskPath in maskList:
            self.listWidget.addItem(maskPath)

    def ComboBoxChanged(self, name ):

        # Change Rendering Mask
        self.imageData.GetPointData().SetActiveAttribute(name, 0)
        self.imageData.GetPointData().Modified()

        # Chagne Rendering Option
        if name == "Scalars_" or name == "ct":
            self.opacityTransferFunction.RemoveAllPoints()
            self.colorTransferFunction.RemoveAllPoints()

            self.opacityTransferFunction.AddPoint(-1024, 0.0)
            self.opacityTransferFunction.AddPoint(3000, 0.5)

            self.colorTransferFunction.AddRGBPoint(0, 0, 0, 0)
            self.colorTransferFunction.AddRGBPoint(1, 1, 1, 1)

            self.mapper.SetBlendModeToMaximumIntensity()
            self.volume.GetProperty().ShadeOff()
        else:
            self.opacityTransferFunction.RemoveAllPoints()
            self.opacityTransferFunction.AddPoint(0, 0.0)
            self.opacityTransferFunction.AddPoint(1, 1.0)

            self.colorTransferFunction.RemoveAllPoints()
            self.colorTransferFunction.AddRGBPoint(0, 0, 0, 0)

            if name == "sk":
                self.colorTransferFunction.AddRGBPoint(1, .23, .52, .64)
            elif name == "mx":
                self.colorTransferFunction.AddRGBPoint(1, .8, 0, 0)
            elif name == "mn":
                self.colorTransferFunction.AddRGBPoint(1, 0, .8, 0)
            elif name == "im":
                self.colorTransferFunction.AddRGBPoint(1, 0, 0, .8)
            else:
                self.colorTransferFunction.AddRGBPoint(1, .79, 0, .21)

            self.mapper.SetBlendModeToComposite()
            self.volume.GetProperty().ShadeOff()

        # Redraw
        self.renWin.Render()

    def ItemClicked(self, item):
        # When Item is clicked, update volume rendering
        dataPath = item.text()

        # Read VTI File
        reader = vtk.vtkXMLImageDataReader()
        reader.SetFileName(dataPath)

        reader.Update()
        self.imageData = reader.GetOutput()

        # ImageData Array
        self.comboBox.clear()
        for i in range(self.imageData.GetPointData().GetNumberOfArrays()):
            self.comboBox.addItem(self.imageData.GetPointData().GetArray(i).GetName())

        self.mapper.SetInputData(self.imageData)

        # Set OTF CTF
        self.ComboBoxChanged("ct")

        self.ren.ResetCamera()
        self.renWin.Render()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='datachecker')
    parser.add_argument('--data_dir', type=str, default="//192.168.0.xxx/Data/vti")
    args = parser.parse_args()

    app = QApplication(["Intraoral"])

    window = Window()
    window.SetDataDir(args.data_dir)

    window.show()

    sys.exit(app.exec_())
