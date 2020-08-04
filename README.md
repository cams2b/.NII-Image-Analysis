# .NII-Image-Analysis
This program compares two .NII images, for similarities. The first image is gathered using a computer algorithm, while the second image is created with a radiologist finding the eyes
This program processes two .NII images, one that utilizes a computer algorithm to find the eyes of a patient and another image where a radiologist has found the eyes of the patient.

The metrics collected by the program are the Intersection Over Union(Jaccard Coefficient), the Dice Coefficient, the Hausdorff distance, the Matthew's Cross Coefficient, and the Accuracy.

This program contains three modes:

1.	Single file processing
  a.	The first mode asks the user to input two file paths. The first file path is the file path to the eyeball-computer results file. The second input is the file   path for the radiologist results file. This mode will output the metrics on these two images into the terminal where the program is being run.
  
2.	Single file processing with image overlay
  a.	This mode takes the same inputs as the first mode. 					  After collecting the file paths, the program will ask for a slice value. If you would prefer the       program to generate a slice value, input 0 and the computer will calculate a slice value. This mode will then output the metrics and display the two images         overlaid in a single image.
  
3.	File Directory processing
    a. This mode will accept two files as input. After taking these arguments, this mode will sort the data and output the metrics for each image pair. After              generating these metrics, this mode will output the data to a .csv file,  and return the average and standard deviation for each metric.
