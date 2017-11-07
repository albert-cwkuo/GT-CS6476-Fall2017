Author: Chia-Wen Kuo, 903319557

Part1:
[ M ] = calculate_projection_matrix( Points_2D, Points_3D )
-Inputs
    Points_2D: nx2 matrix of 2D pixel coordinate on the image
    Points_3D: nx3 matrix of 3D coordinate of corresponding points in the world
-Output
    M: 3x4 projection matrix

[ Center ] = compute_camera_center( M )
-Input
    M: 3x4 projection matrix from the call to calculate_projection_matrix()
-Output
    Center: 3x1 vector of camera center location in the world coordinate
    
Part2:
[ F ] = estimate_fundamental_matrix(Points_a, Points_b, normalize)
-Inputs
    Points_a: nx2 matrix of 2D coordinate of points on Image A
    Points_b: nx2 matrix of 2D coordinate of points on Image B
    normalize: a bool flag indicating whether to normalize the coordinate of correnspondences or not ( extra credits )
-Output
    F: 3x3 fundamental matrix

Part3:
[ Best_Fmatrix, inliers_a, inliers_b] = ransac_fundamental_matrix(matches_a, matches_b)
-Inputs
    matches_a: nx2 matrix of 2D coordinate of points on Image A
    matches_b: nx2 matrix of 2D coordinate of points on Image B
-Output
    Best_Fmatrix: 3x3 fundamental matrix
    inliers_a: mx2 corresponding points on Image A
    inliers_b: mx2 corresponding points on Image B

Extra credits:
    Normalize 2D pixel coordinate for numerical stability. Substract the point coordinate by their mean and divide by their standard deviation such that the points are of zero mean and unit standard deviation.
