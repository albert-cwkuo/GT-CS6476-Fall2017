This is the implementation of SIFT feature based on [1] and [2]. In my implementation, the feature can deal with scale and rotation variance and can achieve 80% accuracy on the difficult Episcopal Gaudi test image.

In order to implement the SIFT, we do not strictly follow the API defined by the placeholder functions. However, you can still run proj2.m to see the results and accuracy.

=================================================
Functional API:
[keypoints, DGS] = get_interest_points(image)
Compute the keypoints of input image. Each keypoint contain (x, y) location, scale, and orientation so that the feature will be invariant to scale and orientation.
- Input:
* image: grey scale image, each pixel is float type ranged from 0~1
- Output:
* DGS: discrete Gaussian space. The down sampled and blurred image pyramid.
* keypoints: arrary of structure. The structure contains several attributes, including: x_dgs: x coordinate in discrete Gaussian space (down sampled Gaussian pyramid),
y_dgs: y coordinate in discrete Gaussian space (down sampled Gaussian pyramid),
s_dgs: scale,
octave: octave,
x_abs: x in the original image coordinate,
y_abs: y in the original image coordinate,
sigma: Gaussian blur sigma in the original image coordinate
ori: orientation of the keypoints.
-------------------------------------------------
keypoints = get_features(keypoints, DGS)
Compute the feature descriptor of each keypoint.
- Input:
* DGS, keypoints: outputs form the last stage(get_interest_points function)
- Output:
* keypoints: keypoint with descriptor attributes. Calculate the feature descriptor and save it in the structure’s feature attribute.
-------------------------------------------------
[matches, confidences] = match_features(keypoints1, keypoints2)
Find the match between keypoints1 and keypoints2
- Input:
* keypoints1, keypoints2: The keypoints and descriptors for image1 and image2.
- Output:
* matches: the match between keypoints1 and keypoints2
* confidences: 1/d of the matched keypoints, in which d is the Euclidean distance between keypoint1 and keypoint2.
-------------------------------------------------
visualizeKeypoints(image, keypoints, name)
Visualize detected keypoints including their location, scale, and orientation
- Input:
* image: the image where the keypoints are detected
* keypoints: detected keypoints with scale and orientation
name: saved filename
- Output: None
-------------------------------------------------
evaluate_correspondence(imgA, imgB, ground_truth_correspondence_file, scale_factor_1, scale_factor_2, x1_est, y1_est, x2_est, y2_est)
Note that I change the API so that image1 and image2 can have different scale_factor. This helps me test the power of scale invariance of my feature.
=================================================

[1] Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints. International journal of computer vision, 60(2), 91-110.

[2] Otero, I. R. (2015). Anatomy of the SIFT Method (Doctoral dissertation, École normale supérieure de Cachan-ENS Cachan).
