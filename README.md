# GenSeg - Generalized Semantic Segmentation

GenSeg is a supervised machine learning model based on SegNet that
semantically segments n-dimensional data.

It can operate on data with N spatial dimensions instead of just two
like in the original SegNet. This enables more complex data, such as
volumetric data, to be used as inputs. However, due to restrictions in
the way that convolutions are implemented in TensorFlow, this
implementation of GenSeg currently works only for 1<=N<=3 spatial
dimensions. Hence, 3D data such as LiDAR is supported, as well as more
conventional data such as images, but higher dimensional data such as 3D time series
data will not be supported until TensorFlow adds support for 4D and
higher convolutions. If/When they do, this architecture can easily then
be made to support those higher dimensions.

The output of GenSeg is a set of class probabilities for each
pixel/voxel/n-dimensional point in the image being segmented. This
allows for entire scenes to be understood similarly to how humans might
understand them.