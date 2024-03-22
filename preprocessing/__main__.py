import argparse
import itk

PixelType = itk.ctype("signed short")
VolumeType = itk.Image[PixelType, 3]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='preprocessing sample')
    parser.add_argument('--sample', type=str, default="//192.168.0.xxx/Data/0001")
    args = parser.parse_args()

    print("Reading Sample Dicom Volume : ", args.sample)
    namesGenerator = itk.GDCMSeriesFileNames.New()
    namesGenerator.SetUseSeriesDetails(True)
    namesGenerator.AddSeriesRestriction("0008|0021")
    namesGenerator.SetDirectory(args.sample)
    seriesUID = namesGenerator.GetSeriesUIDs()
    print("Read Done")

    uid = seriesUID[0]
    fileNames = namesGenerator.GetFileNames(uid)

    reader = itk.ImageSeriesReader[VolumeType].New()
    dicomIO = itk.GDCMImageIO.New()
    reader.SetImageIO(dicomIO)
    reader.SetFileNames(fileNames)
    reader.Update()
    itkImage = reader.GetOutput()

    # Get Dicom TAG
    meta = dicomIO.GetMetaDataDictionary()

    windowCenter = float(meta["0028|1050"])
    windowWidth = float(meta["0028|1051"])
    rescaleIntercept = float(meta["0028|1052"])
    rescaleSlope = float(meta["0028|1053"])

    print(windowCenter, windowWidth, rescaleIntercept, rescaleSlope)
    # #Convert to vtkImageData : itk-vtkglue only available python <=3.7.x
    # itkToVtkFilter = itk.ImageToVTKImageFilter[VolumeType].New()
    # itkToVtkFilter.SetInput(itkImage)
    # itkToVtkFilter.Update()

    # imageData = itkToVtkFilter.GetOuptut()

    # print(imageData)
