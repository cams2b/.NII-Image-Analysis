################################################################################
#                 NII Eyeball Segmentation Performance Metrics                 #
#                                                                              #
# Perform multiple metric analyses of two .nii images. One image is generated  #
# Using a computer algorithm designed to find the eyeball in the scan, while   #
# the second image is captured by a radiologist. This program then computes    #
# the following metrics; IOU, Dice Coefficient, Haufsdorff distance, MCC and   #
# ACC. This program can process single file pairs, or it can compute metrics   #
# for entire files. Furthermore, this program can out put an overlay of the    #
# two images to visualize their overlap.                                       #
#                                                                              #
# Execution: python Eye-Metrics.py                                             #
# Required Libraries: SimpleITK, numpy, matplotlib. sklearn, csv, os, cv2      #
#                                                                              #
#                                                                              #
################################################################################

import SimpleITK as sitk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics
import csv
import os
import cv2
# Global counter for key values


class ImagePair(object):

    def __init__(self, computerPath, radioPath):
        self.computerPath = computerPath
        self.radioPath = radioPath
        self.path = None
        self.computerImage = None
        self.radioImage = None
        self.filename = []
        self.map = []
        self.metrics = []
        self.visuals = []
        self.counter = 0

    # ==============================================================================#
    #                                  Preprocessing                                #
    # ==============================================================================#

    def generate_filename(self, computerPath, radioPath):
        """
        :param computerPath:
        :param radioPath:
        :return: This method returns the longest common substring for the two files
        """
        comp = os.path.split(computerPath)
        radio = os.path.split(radioPath)
        c = comp[1]
        r = radio[1]
        size = min(len(c), len(r))
        i = 0
        filename =''
        while i <= size:
            if c[i] != r[i]:
                return filename
            filename += c[i]
            i += 1
        return filename

    def generate_ctimage(self, ctpath):
        return sitk.ReadImage(ctpath)

    def order_folder(self, comp_dir, rad_dir):
        comp_list = []
        rad_list = []
        for roots, dirs, files in os.walk(comp_dir):
            for file in files:
                filename = comp_dir + "/" + file
                comp_list.append(filename)
        for roots, dirs, files in os.walk(rad_dir):
            for file in files:
                filename = rad_dir + "/" + file
                rad_list.append(filename)
        comp_list.sort()
        rad_list.sort()
        return comp_list, rad_list

    def order_ct(self, ct_dir):
        ct_list = []
        for roots, dirs, files in os.walk(ct_dir):
            for file in files:
                filename = ct_dir + "/" + file
                ct_list.append(filename)
        ct_list.sort()
        return ct_list

    def valid_pair(self, computer, radio):
        if computer.GetDepth() == radio.GetDepth() and computer.GetWidth() == radio.GetWidth() \
                and computer.GetHeight() == radio.GetHeight() \
                and computer.GetSpacing() == radio.GetSpacing():
            return True
        else:
            return False

    def set_slice(self, currimage):
        """
        :param image: SimpleITK image
        :return: This method returns a slice value for image overlay
        """
        val = input("Enter the slice value, or enter 0 to allow the program to generate a slice value: ")
        res = int(val)
        if res == 0:
            print("Generating slice value for current image ...")
            curr = 0
            max = 0
            slice = 0
            for i in range(0, currimage.GetDepth()):
                curr = 0

                for j in range(0, currimage.GetHeight()):
                    for t in range(0, currimage.GetWidth()):

                        val = currimage.GetPixel(t, j, i)
                        if val > 0:
                            curr += 1

                if curr > max:
                    # print("Values: ", i, curr)
                    max = curr
                    slice = i
            return slice
        return res

    def set_output(self):
        path = input("Enter the directory path for data to be exported to: ")
        self.path = path


    # ==============================================================================#
    #                             Performance Metrics                               #
    # ==============================================================================#

    def get_positive_negative(self, comp, radio):
        comparitor = sitk.LabelOverlapMeasuresImageFilter()
        comparitor.Execute(comp, radio)
        Fp = comparitor.GetFalsePositiveError()
        Fn = comparitor.GetFalseNegativeError()
        Tp = 1 - Fn
        Tn = 1 - Fp
        return (Fp, Fn, Tp, Tn)

    def iou(self, comp, radio):
        comparitor = sitk.LabelOverlapMeasuresImageFilter()
        comparitor.Execute(comp, radio)
        return comparitor.GetJaccardCoefficient()


    def dice(self, comp, radio):
        comparitor = sitk.LabelOverlapMeasuresImageFilter()
        comparitor.Execute(comp, radio)
        return comparitor.GetDiceCoefficient()


    def hausdorff(self, comp, radio):
        comparitor = sitk.HausdorffDistanceImageFilter()
        comparitor.Execute(comp, radio)
        return comparitor.GetHausdorffDistance()


    def mcc(self,comp, radio):
        (Fp, Fn, Tp, Tn) = self.get_positive_negative(comp, radio)
        return (Tp * Tn - Fp * Fn) / np.sqrt((Tp + Fp) * (Tp + Fn) * (Tn + Fp) * (Tn + Fn))


    def acc(self, comp, radio):
        (Fp, Fn, Tp, Tn) = self.get_positive_negative(comp, radio)
        return ((Tp + Tn) / (Tp + Tn + Fp + Fn))

    def precision(self, comp, radio):
        (Fp, Fn, Tp, Tn) = self.get_positive_negative(comp, radio)
        return Tp/ (Tp + Fp)
    def sensitivity(self, comp, radio):
        (Fp, Fn, Tp, Tn) = self.get_positive_negative(comp, radio)
        return Tp /(Tp/ Fn)


    def specific_metrics(self, comp, radio, filename, map):
        """

        :param comp: computer image
        :param radio: radiologist image
        :param filename: name of file
        :param map: list of specified metrics
        :return: This method generates the specified metrics for the given image pair and returns
                 the metrics in a list
        """
        print("Processing: ", filename)
        set = []
        set.append(filename)
        for i in map:
            if i == 'iou':
                set.append(self.iou(comp, radio))
            if i == 'dice':
                set.append(self.dice(comp,radio))
            if i == 'hausdorff':
                set.append(self.hausdorff(comp, radio))
            if i == 'mcc':
                set.append(self.mcc(comp, radio))
            if i == 'acc':
                set.append(self.acc(comp, radio))
            if i == 'precision':
                set.append(self.precision(comp, radio))
            if i == 'sensitivity':
                set.append(self.sensitivity(comp, radio))
        self.counter += 1
        print("...... Complete")
        return set
    # ==============================================================================#
    #                               Image Visualization                             #
    # ==============================================================================#

    def overlay(self, comp, radio, ctscan, filename, path, win_min=-1024, win_max=976):
        """
        :param comp: Computer image
        :param radio: Radiologist image
        :param ctscan: Ctscan image
        :param filename: name of file
        :param path: export path
        :param win_min: image parameters
        :param win_max: image parameters
        :return: This method generates an overlaid image containing the computer image and radiologist image.
                 The computer image is overlaid as a green outline, while the radiologist image is overlaid
                 with a red outline
        """

        if self.valid_pair(comp, radio):
            print("Processing ", filename)
            slice = self.set_slice(comp)
            cimage = comp[:, :, slice]
            rimage = radio[:, :, slice]
            ctimage = ctscan[:, :, slice]

            cpath = path + "/" + "computer.png"
            rpath = path + "/" + "radio.png"
            ctpath = path + "/" + filename + '_at_slice_' + str(slice) + ".png"
            # Overlay computer image onto ct scan
            contour_overlaid_image = sitk.LabelMapContourOverlay(sitk.Cast(cimage, sitk.sitkLabelUInt8),
                                                                 sitk.Cast(sitk.IntensityWindowing(ctimage,
                                                                                                   windowMinimum=win_min,
                                                                                                   windowMaximum=win_max),
                                                                           sitk.sitkUInt8), opacity=1,
                                                                 contourThickness=[2, 2])

            sitk.WriteImage(contour_overlaid_image, ctpath)
            sitk.WriteImage(rimage, rpath)

            radio = cv2.imread(rpath, cv2.IMREAD_GRAYSCALE)
            overlay_image = cv2.imread(ctpath)
            # Overlay radiologist image
            contours, _ = cv2.findContours(radio, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for i, c in enumerate(contours):
                mask = np.zeros(radio.shape, np.uint8)
                cv2.drawContours(mask, [c], -1, 255, -1)
                mean, _, _, _ = cv2.mean(radio, mask=mask)
                cv2.drawContours(overlay_image, [c], -1, (0, 0, 255), 1)

            os.remove(rpath)
            os.remove(ctpath)
            #cv2.imwrite(ctpath, overlay_image)
            package = [ctpath, overlay_image]

        return package


    # ==============================================================================#
    #                            User Accessible Methods                            #
    # ==============================================================================#

    def process_folder(self, map):
        """
        This method will process two folders of images and return a 2-dimensional list of
        metrics
        :return:
        """
        self.map = map
        comp_list, rad_list = self.order_folder(self.computerPath, self.radioPath)
        for x, y in zip(comp_list, rad_list):
            self.filename.append(self.generate_filename(x, y))
            currentComp = sitk.ReadImage(x)
            currentRad = sitk.ReadImage(y)
            if self.valid_pair(currentComp, currentRad) == True:

                self.metrics.append(self.specific_metrics(currentComp, currentRad, self.filename[self.counter], map))

    def process_file(self, map):
        """
        Process metrics for an image pair
        :return: list of performance metrics for the image pair
        """
        self.map = map
        self.filename.append(self.generate_filename(self.computerPath, self.radioPath))
        self.computerImage = sitk.ReadImage(self.computerPath)
        self.radioImage = sitk.ReadImage(self.radioPath)

        if self.valid_pair(self.computerImage, self.radioImage) == False:
            print("This image pair cannot be processed due to improper spacing")
            return
        print("This image pair is ready...")
        self.metrics.append(self.specific_metrics(self.computerImage, self.radioImage, self.filename[self.counter], map))

    def visualize_image(self, ctpath):
        """
        :param ctpath: Path to original CT scan
        :return: This method overlays the computer and radiologist image onto the oringal CT scan.
        """
        if len(self.filename) == 0:
            print('empty')
        self.computerImage = sitk.ReadImage(self.computerPath)
        self.radioImage = sitk.ReadImage(self.radioPath)
        ctimage = sitk.ReadImage(ctpath)
        self.set_output()

        self.visuals.append(self.overlay(self.computerImage, self.radioImage, ctimage, self.filename[0], self.path))

    def visualize_folder(self, ctpath):
        """
        :param ctpath: CT scan path
        :return: This method visualizes all files contained within the directories and stores them
                 in the visuals list.
        """
        comp_list, rad_list,  = self.order_folder(self.computerPath, self.radioPath)
        # If the filenames have not been added -> add
        if len(self.filename) == 0:
            for x, y in zip(comp_list, rad_list):
                self.filename.append(self.generate_filename(x, y))
        ct_list = self.order_ct(ctpath)
        self.set_output()
        i = 0
        for x, y, z in zip(comp_list, rad_list, ct_list):
            a = sitk.ReadImage(x)
            b = sitk.ReadImage(y)
            c = sitk.ReadImage(z)
            self.visuals.append(self.overlay(a, b, c, self.filename[i], self.path))
            i += 1

    def export_metrics(self):
        """
        This method generates a .csv spreadsheet with all gathered metrics. For each
        metric, the standard deviation and average is returned in the bottom two columns
        :return: metrics.csv exported to designated path
        """
        self.set_output()
        inScan = self.path + "/" + "metrics.csv"
        if len(self.metrics) == 0:
            print("No metrics have been gathered.")
            return
        columns = ['filename']
        for i in self.map:
            columns.append(i)
        df = pd.DataFrame(data=self.metrics, columns=columns)

        std = df.std()
        avg = df.mean()
        arr1 = ['Std Dev.']
        arr2 = ['Average']
        for i, y in zip(std, avg):
            arr1.append(i)
            arr2.append(y)
        self.metrics.append(arr1)
        self.metrics.append(arr2)
        df2 = pd.DataFrame(data=self.metrics, columns=columns)
        df2.to_csv(inScan)
        print("Metrics.csv exported to ", self.path)

    def export_visuals(self):
        """
        This method exports all images to the designated directory
        """

        if len(self.visuals) == 0:
            print("There are no images to export")
            return
        p = os.path.split(self.visuals[0][0])
        p = p[0]
        for i in self.visuals:
            cv2.imwrite(i[0], i[1])
        print("Images exported to: ", p)



def main():
    cfile = "/Users/cameronbeeche/Desktop/Research/imageAnalysis/Test/eyeball-computer-results/HB039126OAV_00230_2014-03-29_2_img.nii"
    rfile = "/Users/cameronbeeche/Desktop/Research/imageAnalysis/Test/msk-eyeball-radiologist-results/HB039126OAV_00230_2014-03-29_2_msk.nii"
    ctfile = '/Users/cameronbeeche/Desktop/Research/imageAnalysis/Test/img-eyeball/HB039126OAV_00230_2014-03-29_2_img.nii'
    cfolder = '/Users/cameronbeeche/Desktop/Research/imageAnalysis/Test/eyeball-computer-results'
    rfolder = '/Users/cameronbeeche/Desktop/Research/imageAnalysis/Test/msk-eyeball-radiologist-results'
    ctfolder = '/Users/cameronbeeche/Desktop/Research/imageAnalysis/Test/img-eyeball'

    map = ['iou', 'dice', 'hausdorff', 'mcc', 'acc', 'precision', 'sensitivity']
    map2 = ['iou', 'dice', 'hausdorff', 'mcc', 'acc']

    test = ImagePair(cfile, rfile)
    test2 = ImagePair(cfolder, rfolder)

    test.process_file(map)
    test.visualize_image(ctfile)
    test.export_visuals()


    test2.process_folder(map2)
    test2.export_metrics()

    test2.visualize_folder(ctfolder)
    test2.export_visuals()


if __name__ == "__main__":
    main()