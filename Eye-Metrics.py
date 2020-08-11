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
import matplotlib.pyplot as plt
import sklearn.metrics
import csv
import os
import cv2
# Global counter for key values
counter = 0

def main():
    print("Eye segmentation metrics")
    mode = input("1. File 2. Directory ")
    if mode == '1':
        print("File processing")
        computerpath = input("Enter file path for computer image: ")
        radiopath = input("Enter file path for radiologist image: ")
        CTpath = input("Enter file path for CT scan: ")
        output = input("Enter directory path for metrics to be exported to: ")
        singleImage = ImagePair(computerpath, radiopath, CTpath, output, 'file')
        singleImage.metric_overlay()
    if mode == '2':
        print("Directory processing")
        computerdir = input("Enter the computer directory: ")
        radiodir = input("Enter the radiologist directory: ")
        CTdir = input("Enter the CT directory: ")
        output = input("Enter the directory path for metrics to be exported to: ")
        dirComp = ImagePair(computerdir, radiodir, CTdir, output, 'dir')
        dirComp.dir_output()
    return

########################################
# Arguments: SimpleITK image object
# If not given a slice value, this method
# will calculate a slice value.
########################################
def calc_slice(currimage):
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

########################################
# Arguments: Two SimpleITK image object
# This method will determine if both images
# have equal dimensions and spacing.
########################################
def equal_dimensions(comp, radio):

    if comp.GetDepth() == radio.GetDepth() and comp.GetWidth() == radio.GetWidth() \
                                           and comp.GetHeight() == radio.GetHeight() \
                                           and comp.GetSpacing() == radio.GetSpacing():
        return True
    else:
        return False

########################################
# Arguments: 3 file directories and 3 lists
# This method will order each directory into
# a list where there position corresponds
# to the computer and radiologist image
########################################
def directory_processing(comp_dir, rad_dir, ct_dir, comp_list, rad_list, ct_list):
    # Iterate through both directories and add the values to their respective lists
    for roots, dirs, files in os.walk(comp_dir):
        for file in files:
            filename = comp_dir + "/" + file
            comp_list.append(filename)
    for roots, dirs, files in os.walk(rad_dir):
        for file in files:
            filename = rad_dir + "/" + file
            rad_list.append(filename)
    for roots, dirs, files in os.walk(ct_dir):
        for file in files:
            filename = ct_dir + "/" + file
            ct_list.append(filename)
    ct_list.sort()
    comp_list.sort()
    rad_list.sort()
    return

def set_slice(image):
    var = input("Enter slice value, or enter 0 to allow the computer to generate a slice value: ")
    var = int(var)
    if var == 0:
        var = calc_slice(image)
    return var


def get_slice(image, slice):

    return image[:, :, slice]






########################################
# Arguments:  file path,
# This method isolates the filename from the filepath.
#
########################################
def filename(path):
    row = []
    fname = path.split('/')
    alter = fname[len(fname) - 1]
    alter = alter[0: len(alter) - 7]

    return alter


def file_output(comp, radio):
    print("IOU: ", iou(comp, radio))
    print("Dice: ", dice(comp, radio))
    print("Hausdorff: ", hausdorff(comp, radio))
    print("MCC: ", mcc(comp, radio))
    print("ACC: ", acc(comp, radio))

def dir_output(comp_dir, rad_dir):
    output = []
    for x, y in zip(comp_dir, rad_dir):
        c = sitk.ReadImage(x)
        r = sitk.ReadImage(y)
        if equal_dimensions(c, r):
            output.append(csv_row(c, r, x))



    return output


########################################
# Arguments: computer file path, radiologist file path
# This method returns a single entry for directory output
# A directory output contains the following:
# 1. key 2. filename 3. IOU 4. Dice Coefficient
# 5. Hausdorff 6. Matthew's Correlation Coefficient
# 7. Accuracy
#
########################################
def csv_row(comp, radio, cpath):
    row = []
    fname = cpath.split('/')
    global counter
    counter += 1
    row.append(counter)
    row.append(fname[len(fname) - 1])
    # Create sitk images

    row.append(iou(comp, radio))
    row.append(dice(comp, radio))
    row.append(hausdorff(comp, radio))
    row.append(mcc(comp, radio))
    row.append(acc(comp, radio))
    return row


########################################
# Arguments: 2 Dimensional list
# This method exports directory metrics
# to a .csv file
########################################
def export(data, path):

    inScan = path + "/" + "metrics.csv"
    with open(inScan, 'w', newline='') as csvfile:
        fieldnames = ['key', 'filename', 'IOU', 'Dice_Coefficient', 'Hausdorff', 'MCC', 'ACC']

        # Dictionary writer.
        thewriter = csv.DictWriter(csvfile, fieldnames=fieldnames)

        thewriter.writeheader()

        for disk in data:
            thewriter.writerow({'key': disk[0], 'filename': disk[1], 'IOU': disk[2],
                                'Dice_Coefficient': disk[3], 'Hausdorff': disk[4],
                                'MCC': disk[5], 'ACC': disk[6]})


    return

########################################
# Accepts metrics and returns STD Dev
# and Average in the form of an appended
# CSV row
########################################
def calc_avg_std(data):
    tiou = []
    tdice = []
    thaus = []
    tmcc = []
    tacc = []

    avgStats = []
    stdStats = []
    avgStats.append("Average")
    avgStats.append(0)
    stdStats.append("Std Dev")
    stdStats.append(0)
    for i in data:
        tiou.append(i[2])
        tdice.append(i[3])
        thaus.append(i[4])
        tmcc.append(i[5])
        tacc.append(i[6])
    # Calculate averages and std dev for each metric.
    avgStats.extend((Average(tiou), Average(tdice), Average(thaus), Average(tmcc), Average(tacc)))
    stdStats.extend((np.std(tiou), np.std(tdice), np.std(thaus), np.std(tmcc), np.std(tacc)))
    data.extend((avgStats, stdStats))
    return data

########################################
# Arguments: list
# This method calculates the average
########################################
def Average(arr):

    return sum(arr) / len(arr)

########################################
# Arguments: SimpleITK computer scan, SimpleITk radiologist scan
# This method returns the intersection over union (Jaccard Coefficient)
#
########################################
def iou(comp, radio):
    comparitor = sitk.LabelOverlapMeasuresImageFilter()
    comparitor.Execute(comp, radio)
    return comparitor.GetJaccardCoefficient()

########################################
# Arguments: SimpleITK computer scan, SimpleITk radiologist scan
# This method returns the Dice Coefficient
#
########################################
def dice(comp, radio):
    comparitor = sitk.LabelOverlapMeasuresImageFilter()
    comparitor.Execute(comp, radio)
    return comparitor.GetDiceCoefficient()

########################################
# Arguments: SimpleITK computer scan, SimpleITk radiologist scan
# This method returns the Hausdorff Coefficient
#
########################################
def hausdorff(comp, radio):
    comparitor = sitk.HausdorffDistanceImageFilter()
    comparitor.Execute(comp, radio)
    return comparitor.GetHausdorffDistance()

########################################
# Arguments: SimpleITK computer scan, SimpleITk radiologist scan
# This method returns the Matthews Correlation Coefficient
#
########################################
def mcc(comp, radio):
    comparitor = sitk.LabelOverlapMeasuresImageFilter()
    comparitor.Execute(comp, radio)
    Fp = comparitor.GetFalsePositiveError()
    Fn = comparitor.GetFalseNegativeError()
    Tp = 1 - Fn
    Tn = 1 - Fp
    return (Tp * Tn - Fp * Fn) / np.sqrt((Tp + Fp) * (Tp + Fn) * (Tn + Fp) * (Tn + Fn))

########################################
# Arguments: SimpleITK computer scan, SimpleITk radiologist scan
# This method returns the Accuracy between the two images
#
########################################
def acc(comp, radio):
    comparitor = sitk.LabelOverlapMeasuresImageFilter()
    comparitor.Execute(comp, radio)
    Fp = comparitor.GetFalsePositiveError()
    Fn = comparitor.GetFalseNegativeError()
    Tp = 1 - Fn
    Tn = 1 - Fp
    return ((Tp + Tn) / (Tp + Tn + Fp + Fn))

########################################
# Arguments: SimpleITK computer image file path, SimpleITk computer image,
# SimpleITk radiologist scan, SimpleITk computer scan, output path, slice
# This overlays the computer and radiologist image on the original CT
# scan
########################################
def overlay(tpath, comp, radio, ctscan, path, slice, win_min=-1024, win_max=976):

    if equal_dimensions(comp, radio):
        cimage = get_slice(comp, slice)
        rimage = get_slice(radio, slice)
        orgimage = get_slice(ctscan, slice)
        cpath = path + "/" + "computer.png"
        rpath = path + "/" + "radio.png"
        ctpath = path + "/" + filename(tpath) + ".png"
        # Overlay computer image onto ct scan
        contour_overlaid_image = sitk.LabelMapContourOverlay(sitk.Cast(cimage, sitk.sitkLabelUInt8),
                                                             sitk.Cast(sitk.IntensityWindowing(orgimage, windowMinimum=win_min,windowMaximum=win_max),
                                                             sitk.sitkUInt8), opacity=1, contourThickness=[2, 2])

        sitk.WriteImage(contour_overlaid_image, ctpath)
        sitk.WriteImage(rimage, rpath)
        CT = cv2.imread(ctpath)
        radio = cv2.imread(rpath, cv2.IMREAD_GRAYSCALE)
        # Overlay radiologist image
        contours, _ = cv2.findContours(radio, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for i, c in enumerate(contours):
            mask = np.zeros(radio.shape, np.uint8)
            cv2.drawContours(mask, [c], -1, 255, -1)
            mean, _, _, _ = cv2.mean(radio, mask=mask)
            cv2.drawContours(CT, [c], -1, (0, 0, 255), 1)

        os.remove(rpath)
        os.remove(ctpath)
        cv2.imwrite(ctpath, CT)
        print("Image overlay complete " + filename(tpath) + " exported to: ", path)
    return


########################################
# Arguments: computer image path, radiologist image path, type (either file or dir)
# Creates an image pair object for metrics to be performed upon
#
#
########################################
class ImagePair(object):

    # Initialize object
    def __init__(self, cpath, rpath, CTpath, path, typ):

        self.cpath = cpath
        self.rpath = rpath
        self.path = path
        self.cimage = None
        self.rimage = None
        self.ctimage = None
        self.metrics = []
        self.comp_list = []
        self.rad_list = []
        self.ct_list = []
        self.slice = 999


        if typ == 'file':
            self.typ = 'file'
            self.cimage = sitk.ReadImage(cpath)
            self.rimage = sitk.ReadImage(rpath)
            self.ctimage = sitk.ReadImage(CTpath)
            if self.slice == 999:
                self.slice = set_slice(self.cimage)
            self.cslice = get_slice(self.cimage, self.slice)
            self.rslice = get_slice(self.rimage, self.slice)
            self.ctslice = get_slice(self.ctimage, self.slice)


        if typ == 'dir':
            self.typ = 'dir'
            directory_processing(cpath, rpath, CTpath, self.comp_list, self.rad_list, self.ct_list)


    def img_overlay(self):
        overlay(self.cpath, self.cimage, self.rimage, self.ctimage, self.path, self.slice)

    def iou(self):
        return iou(self.cimage, self.rimage)

    def dice(self):
        return dice(self.cimage, self.rimage)

    def hausdorff(self):
        return hausdorff(self.cimage, self.rimage)

    def mcc(self):
        return mcc(self.cimage, self.rimage)

    def acc(self):
        return acc(self.cimage, self.rimage)

    def metric_overlay(self):
        print("Generating metrics for: ", filename(self.cpath))
        file_output(self.cimage, self.rimage)
        overlay(self.cpath, self.cimage, self.rimage, self.ctimage, self.path, self.slice)
        print("Complete. Metrics exported to: ", self.path)
        overlay(self.cpath, self.cimage, self.rimage, self.ctimage, self.path, self.slice)

    def file_output(self):
        print("Generating metrics for: ", filename(self.cpath))
        file_output(self.cimage, self.rimage)
        overlay(self.cpath, self.cimage, self.rimage, self.ctimage, self.path, self.slice)

    def dir_output(self):
        if self.typ == 'file':
            print("This object contained file paths, not directories")
            return
        print("Generating metrics ...")
        self.metrics = dir_output(self.comp_list, self.rad_list)
        self.metrics = calc_avg_std(self.metrics)
        export(self.metrics, self.path)
        print("Complete. Metrics exported to: ", self.path)

if __name__ == "__main__":

    main()

