import SimpleITK as sitk
import os
import numpy as np
import sklearn.metrics
import csv
import cv2
import matplotlib.pyplot as plt
import numpy.ma as ma
from tkinter import *
from tkinter import simpledialog
from tkinter import filedialog


comp_dir = "/Users/cameronbeeche/Desktop/Research/imageAnalysis/Test/eyeball-computer-results"
rad_dir = "/Users/cameronbeeche/Desktop/Research/imageAnalysis/Test/msk-eyeball-radiologist-results"

comp_list = []
rad_list = []
top = 0
bot = 0

Tp = 0
Tn = 0
Fp = 0
Fn = 0

s = ''
cf = ''
rf = ''

# Record counter
diskCount = 0
inScan = 'metrics.csv'

def main():


    root = Tk()
    root.geometry("300x300")
    root.withdraw()
    s = simpledialog.askstring(title="1. Print individual file data 2. Print file data and visual overlay 3. Directory",
                                prompt="select:")

    if s == '1':
        f1 = simpledialog.askstring(title="file1",
                                    prompt="Computer Filepath:")
        f2 = simpledialog.askstring(title="file2",
                                    prompt="Radiologist Filepath:")

        process(f1, f2)



    if s == '2':
        print("Please input entire file directory if the files are not in the directory of this program")
        #f1 = input("Input computer generated image filename: ")
        #print(f1)
        #f2 = input("Input radiologist image filename: ")
        #print(f2)

        master = Tk()
        master.withdraw()
        f1 = simpledialog.askstring(title="file1", prompt="Computer Filepath:")
        f2 = simpledialog.askstring(title="file2", prompt="Radiologist Filepath:")




        print("Input the given slice value for the image display. If you want the program to generate this value for you please input 0")

        slice = simpledialog.askinteger(title="Slice",
                                    prompt="Input slice :")



        #global cf

        process(f1, f2)

        cimage = sitk.ReadImage(f1)
        rimage = sitk.ReadImage(f2)

        if cimage.GetDepth() != rimage.GetDepth():

            print("These files have different dimensions, please input files with equal dimensions")
            return

        curr = 0
        max = 0

        #slice
        if slice == 0:
            # Find slice of Nifti image
            for i in range(0, cimage.GetDepth()):
                curr = 0

                for j in range(0, cimage.GetHeight()):
                    for t in range(0, cimage.GetWidth()):

                        val = cimage.GetPixel(t, j, i)
                        if val > 0:
                            curr += 1

                if curr > max:
                    # print("Values: ", i, curr)
                    max = curr
                    slice = i




        # convert to array
        ver1 = sitk.GetArrayFromImage(cimage)
        ver2 = sitk.GetArrayFromImage(rimage)
        A = ver1[slice, :, :]
        B = ver2[slice, :, :]
        cv2.imwrite("computer_2d_slice.jpg", A)
        cv2.imwrite("radiologist_2d_slice.jpg", B)

        c2d = sitk.ReadImage("computer_2d_slice.jpg")
        r2d = sitk.ReadImage("radiologist_2d_slice.jpg")

        RAR = sitk.GetArrayFromImage(c2d)
        RAR2 = sitk.GetArrayFromImage(r2d)

        mask = ma.masked_where(RAR2 > 0, RAR)
        Image2_mask = ma.masked_array(RAR2, mask)

        plt.imshow(RAR, cmap='Reds')
        plt.imshow(Image2_mask, cmap='Blues', alpha=0.5)
        plt.title('Image Overlay')

        plt.show()




    if s == '3':

        #comp_dir = simpledialog.askstring(title="Test", prompt="Computer Filepath:")
        #rad_dir = simpledialog.askstring(title="Test", prompt="Radiologist Filepath:")

        accept = Tk()

        comp_dir = filedialog.askdirectory(parent=accept,
                                         initialdir=os.getcwd(),
                                         title="Please select computer algr folder:")

        rad_dir = filedialog.askdirectory(parent=accept,
                                         initialdir=os.getcwd(),
                                         title="Please select radiologist folder:")

        # Iterate through both directories and add the values to their respective lists
        for roots, dirs, files in os.walk(comp_dir):
            for file in files:

                #print("File = %s" % file)
                filename = comp_dir + "/" + file
                #curr = sitk.ReadImage(filename)
                comp_list.append(filename)

        for roots, dirs, files in os.walk(rad_dir):
            for file in files:

                #print("File = %s" % file)
                filename = rad_dir + "/" + file
                #curr = sitk.ReadImage(filename)
                rad_list.append(filename)

        #print(len(comp_list))
        #print(len(rad_list))

        # Sort the lists
        comp_list.sort()
        rad_list.sort()



        # Create array of records
        records = []

        totalIOU = []
        totalDice = []
        totalHaus = []
        totalMCC = []
        totalACC = []

        avgStats = []
        stdStats = []
        avgStats.append('Average')
        avgStats.append('')
        stdStats.append('STD DEV')
        stdStats.append('')



        # Main loop

        for x, y in zip(comp_list, rad_list):
            print("==========")
            print(x)  # X is comp
            print(y)  # Y is rad
            currVal = process(x, y)

            if currVal[2] != 0:
                totalIOU.append(currVal[2])
                totalDice.append(currVal[3])
                totalHaus.append(currVal[4])
                totalMCC.append(currVal[5])
                totalACC.append(currVal[6])

            records.append(currVal)

        avgStats.append(Average(totalIOU))
        avgStats.append(Average(totalDice))
        avgStats.append(Average(totalHaus))
        avgStats.append(Average(totalMCC))
        avgStats.append(Average(totalACC))


        stdStats.append(np.std(totalIOU))
        stdStats.append(np.std(totalDice))
        stdStats.append(np.std(totalHaus))
        stdStats.append(np.std(totalMCC))
        stdStats.append(np.std(totalACC))

        records.append(avgStats)
        records.append(stdStats)




        # Write to csv
        with open(inScan, 'w', newline='') as csvfile:

            fieldnames = ['key', 'filename', 'IOU', 'Dice_Coefficient', 'Hausdorff', 'MCC', 'ACC']

            # Dictionary writer.
            thewriter = csv.DictWriter(csvfile, fieldnames=fieldnames)

            thewriter.writeheader()


            for disk in records:
                thewriter.writerow({'key':disk[0], 'filename':disk[1], 'IOU':disk[2],
                                    'Dice_Coefficient':disk[3], 'Hausdorff':disk[4],
                                    'MCC':disk[5], 'ACC':disk[6]})




    return

def Average(arr):

    return sum(arr) / len(arr)

def process(a, b):

    # Create return array
    disk = []

    global diskCount
    diskCount += 1

    # Append counter and f name
    disk.append(diskCount)
    disk.append(a)

    # Convert values into images
    comp = sitk.ReadImage(a)
    rad = sitk.ReadImage(b)

    enter = False
    if rad.GetDepth() == comp.GetDepth() and rad.GetSpacing() == comp.GetSpacing():
        comparitor = sitk.LabelOverlapMeasuresImageFilter()
        comparitor.Execute(comp, rad)

        enter = True


        jac = comparitor.GetJaccardCoefficient()
        disk.append(jac)
        print("IOU: ", jac)

        dice = comparitor.GetDiceCoefficient()
        disk.append(dice)
        print("Dice Coefficient: ", dice)


        similarity = sitk.SimilarityIndexImageFilter()
        similarity.Execute(rad, comp)


        distance = sitk.HausdorffDistanceImageFilter()
        distance.Execute(rad, comp)

        dorff = distance.GetHausdorffDistance()
        disk.append(dorff)
        print("Hausdorff Distance: ", dorff)

        # Get pixels in an array
        radArr = sitk.GetArrayFromImage(rad)
        compArr = sitk.GetArrayFromImage(comp)

        # Convert array into numpy array
        rar = np.array(radArr)
        car = np.array(compArr)
        # Flatten array into 1d
        r = rar.flatten()
        c= car.flatten()

        #sklearn.metrics.confusion_matrix(r, c, labels=None, sample_weight=None, normalize=None)
        mcc = sklearn.metrics.matthews_corrcoef(r, c, sample_weight=None)
        disk.append(mcc)
        print("MCC: ", mcc)

        accurate = sklearn.metrics.accuracy_score(r, c, normalize=True, sample_weight=None)
        disk.append(accurate)
        print("ACC: ", accurate)



    if enter == False:
        disk.append(0)
        disk.append(0)
        disk.append(0)
        disk.append(0)
        disk.append(0)

    return disk
        # Export data to metrics.csv



def get_mode():
    global s
    s = simpledialog.askstring("Input String", "1. Individual file 2. Directory")

def get_cfile():
    global cf
    cf = simpledialog.askstring(("Input the computer algorithm file path: "))

def get_ffile():
    global ff
    ff =simpledialog.askstring("Input the radiologist file path: ")

if __name__ == "__main__":
    main()



