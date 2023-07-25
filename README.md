# Image-Analysis-of-3D-Printed-Scaffolds-
Within this repository are the Jupyter notebook and the needed files to run this image analysis pipeline to characterize images of 3D printed scaffold's pore size and fiber size. 

## Table of Content
1. Scaffold Analysis.ipynb: Jupyter notebook that is the interface of the pipeline
2. GNP_scaffold_ia.py: Python file that contains the functions that are called in the Scaffold Analysis.ipynb
3. Example Scaffold.tif: Example image of a processed image that will be measured
4. Exampled Scale.png: Scale image to calibrate pixel to mm for the example scaffolds. Image of 10 mms 
5. Example Scale_2.png: Scale image to calibrate pixel to mm for the example image folder. Image of 10 mms 
6. Example Images: Folder of 4 scaffold images
7. Example Images_Six Fiber Scaffolds.csv: Produced Excel sheet with scaffold measurements that had six fibers along one direction
8. Example Images_Other Fibers Scaffolds.csv: Produced Excel sheet with scaffold measurements that did not have six fibers along a direction. For names of the column, manual reference to the Jupyter notebook is needed for these measurements

## Instructions
1. Replace example images with images of interest. This repository shows the bare minimum needed to run the entire notebook. 
2. Open Scaffold Analysis.ipynb within the same folder as the images and change the path names to match the images of interest.
    1. Under "Running Method on One Image"
       * Change "path = " to name of location of image
       * Change "scale = (" to the name of the scale's location
    2. Under "Running a Folder of Images"
       * Change "directory = " to the name of the folder of images
       * Change "scale = (" to the name of the scale's location
3. Once the entire notebook runs, two excel sheets will be created in the same folder as the notebook that compile the data generated from running an entire folder of image. 
