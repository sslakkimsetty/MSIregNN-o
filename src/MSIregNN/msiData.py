import numpy as np
import matplotlib.pyplot as plt


class MSIData():
    """
    Class for handling Mass Spectrometry Imaging (MSI) data.

    :Parameters:
        - fpath_ints (str): Filepath for the intensity data (e.g., CSV file).
        - fpath_pixels (str): Filepath for the pixel data (e.g., CSV file).
        - fpath_features (str): Filepath for the feature data (e.g., CSV file).

    :Attributes:
        - ints (np.ndarray): Intensity data loaded from 'fpath_ints'.
        - pixels (np.ndarray): Pixel data loaded from 'fpath_pixels'.
        - mz (np.ndarray): Feature data loaded from 'fpath_features'.
        - nFeatures (int): Number of features in the data.
        - areIntensitiesBuilt (bool): Flag indicating whether intensity matrix is built.
        - nX (int): Number of pixels along the x-axis.
        - nY (int): Number of pixels along the y-axis.
        - intensities (np.ndarray): Intensity matrix of shape (nFeatures, nX, nY).

    :Methods:
    - `__init__(fpath_ints, fpath_pixels, fpath_features)`: Constructor method.
    - `showPixelLoc(quietly=False) -> np.ndarray`: Display the pixel locations on a plot.
    - `buildIntensityMatrix() -> np.ndarray`: Build the intensity matrix from the loaded data.
    - `findNearestFeature(mz) -> int`: Find the index of the nearest feature in mz.
    - `image(mz, **kwargs)`: Display the ion image for a given mz.
    - `exportIonImages(mz) -> np.ndarray`: Export ion images for specified mz values.
    """
    def __init__(self, fpath_ints, fpath_pixels, fpath_features):
        self.ints = np.loadtxt(fpath_ints, dtype="str", delimiter=",")
        self.ints = self.ints[1:, 1:].astype(np.float32)

        self.pixels = np.loadtxt(fpath_pixels, dtype="str", delimiter=",")
        self.pixels = self.pixels[1:, 1:].astype(np.int32).T
        self.pixels[0] = self.pixels[0] - np.min(self.pixels[0])
        self.pixels[1] = self.pixels[1] - np.min(self.pixels[1])

        self.mz = np.loadtxt(fpath_features, dtype="str", delimiter=",")
        self.mz = self.mz[1:, 1].astype(np.float32)

        self.nFeatures = len(self.mz)
        self.areIntensitiesBuilt = False
        _ = self.showPixelLoc(quietly=True)

    def showPixelLoc(self, quietly=False):
        """
        Display the pixel locations on a plot.

        :param quietly: If True, suppress the plot display.
        :return: Array representing pixel locations.
        """
        pixels_present = np.zeros((np.max(self.pixels[1]+1), np.max(self.pixels[0]+1))).astype(np.float32)
        pixels_present[self.pixels[1], self.pixels[0]] = 1.0

        self.nX = pixels_present.shape[0]
        self.nY = pixels_present.shape[1]

        if not quietly:
            plt.imshow(pixels_present, cmap=plt.cm.binary)
        return pixels_present

    def buildIntensityMatrix(self):
        """
        Build the intensity matrix from the loaded data.

        :return: Intensity matrix of shape (nFeatures, nX, nY).
        """
        self.intensities = np.zeros((self.nFeatures, self.nX, self.nY)).astype(np.float32)
        self.intensities[:, self.pixels[1], self.pixels[0]] = self.ints
        self.areIntensitiesBuilt = True
        return self.intensities

    def findNearestFeature(self, mz):
        """
        Find the index of the nearest feature in mz.

        :param mz: Mass-to-charge ratio.
        :return: Index of the nearest feature.
        """
        idx = (np.abs(self.mz - mz)).argmin()
        return idx

    def image(self, mz, **kwargs):
        """
        Display the ion image for a given mz.

        :param mz: Mass-to-charge ratio.
        :param kwargs: Additional keyword arguments for plt.imshow.
        """
        if not self.areIntensitiesBuilt:
            _ = self.buildIntensityMatrix()

        idx = self.findNearestFeature(mz)
        plt.imshow(self.intensities[idx], **kwargs)

    def exportIonImages(self, mz):
        """
        Export ion images for specified mz values.

        :param mz: List of mass-to-charge ratios.
        :return: Ion images for specified mz values.
        """
        idx = list(map(self.findNearestFeature, mz))
        idx.sort()
        return self.intensities[idx]
