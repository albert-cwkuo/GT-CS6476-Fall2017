% Projection Matrix Stencil Code
% CS 4476 / 6476: Computer Vision, Georgia Tech
% Written by Henry Hu, Grady Williams, James Hays

% Returns the projection matrix for a given set of corresponding 2D and
% 3D points. 

% 'Points_2D' is nx2 matrix of 2D coordinate of points on the image
% 'Points_3D' is nx3 matrix of 3D coordinate of points in the world

% 'M' is the 3x4 projection matrix


function [ M ] = calculate_projection_matrix( Points_2D, Points_3D )

% To solve for the projection matrix. You need to set up a system of
% equations using the corresponding 2D and 3D points:

%                                                     [M11       [ u1
%                                                      M12         v1
%                                                      M13         .
%                                                      M14         .
%[ X1 Y1 Z1 1 0  0  0  0 -u1*X1 -u1*Y1 -u1*Z1          M21         .
%  0  0  0  0 X1 Y1 Z1 1 -v1*X1 -v1*Y1 -v1*Z1          M22         .
%  .  .  .  . .  .  .  .    .     .      .          *  M23   =     .
%  Xn Yn Zn 1 0  0  0  0 -un*Xn -un*Yn -un*Zn          M24         .
%  0  0  0  0 Xn Yn Zn 1 -vn*Xn -vn*Yn -vn*Zn ]        M31         .
%                                                      M32         un
%                                                      M33         vn ]

% Then you can solve this using least squares with the '\' operator or SVD.
% Notice you obtain 2 equations for each corresponding 2D and 3D point
% pair. To solve this, you need at least 6 point pairs.

%%%%%%%%%%%%%%%%
% Your code here
%%%%%%%%%%%%%%%%

% fprintf('Randomly setting matrix entries as a placeholder\n')
% M = [0.1768    0.7018    0.7948    0.4613; ...
%      0.6750    0.3152    0.1136    0.0480; ...
%      0.1020    0.1725    0.7244    0.9932];
%This M matrix came from a call to rand(3,4). It leads to a high residual.
%Your total residual should be less than 1.
num_pts = size(Points_2D, 1);

A=zeros(num_pts*2, 11);
V=zeros(num_pts*2, 1);

for i=1:num_pts
    x = Points_3D(i,1);
    y = Points_3D(i,2);
    z = Points_3D(i,3);
    u = Points_2D(i,1);
    v = Points_2D(i,2);
    
    A(2*i-1,:)=[x y z 1 0 0 0 0 -u*x -u*y -u*z];
    A(2*i,:)=[0  0  0  0 x y z 1 -v*x -v*y -v*z];
    
    V(2*i-1,1)=u;
    V(2*i,1)=v;
end

m = A\V;
m(end+1,1)=1;
M = reshape(m, [4 3])';
M=M;

end

