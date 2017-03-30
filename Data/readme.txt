Aug 12, 2015

The attached files contain ground truth labels data used in the following paper.

sensor fusion for Semantic Segmentation of Urban Scenes
Richard Zhang, Stefan A. Candra, Kai Vetter, Avideh Zakhor
In ICRA, 2015
http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7139439

Segmentation annotations are added to the **tracking training** dataset in the KITTI dataset. Download the raw data from http://www.cvlibs.net/datasets/kitti/.

A total of 8 runs are used, with every 10th or 20th acquisition labeled. A total of 252 acquisitions are labeled across the 8 runs. For the paper, train/test splits are as follows:
 - 140 acquisitions from runs 9, 10, 11, 19 used for training
 - 112 acquisitions from runs 0, 4, 5, 13 used for testing

Both images and velodyne points are ground truthed. Image ground truth labels are saved as Matlab binaries under ./image_02/['trn' or 'test']/[run number]/[acquisition number]. Images of the ground truth annotations are provided as well. Loading the Matlab binaries provides a matrix of integers. The corresponding object classes, along with RGB values in the images are below.

Label number / Object class / RGB
0 - NOT GROUND TRUTHED - 255 255 255
1 - building - 153 0 0
2 - sky - 0 51 102
3 - road - 160 160 160
4 - vegetation - 0 102 0
5 - sidewalk - 255 228 196
6 - car - 255 200 50
7 - pedestrian - 255 153 255
8 - cyclist - 204 153 255
9 - signage - 130 255 255
10 - fence - 193 120 87

Velodyne ground truth labels are saved as Matlab binaries under ./velodyne/['trn' or 'test']/[run number]/[acquisition  number]. Note that velodyne ground truth labels were done separately from the images, ie they are not obtained through image to point cloud projections.

For any questions, please contact Richard Zhang at rich.zhang@eecs.berkeley.edu.