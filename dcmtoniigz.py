import dicom2nifti as dcmc
import os 

input_dir = r"C:\Users\hslab\OneDrive\Desktop\MANUAL\final_lung\manifest-1600709154662\LIDC-IDRI\LIDC-IDRI-0001\01-01-2000-NA-NA-30178\3000566.000000-NA-03192"
output_dir=r"C:\Users\hslab\OneDrive\Desktop\MANUAL\output"


dcmc.convert_directory(input_dir,output_dir)



