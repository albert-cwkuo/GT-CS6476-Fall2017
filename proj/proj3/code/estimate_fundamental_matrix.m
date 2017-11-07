% Fundamental Matrix Stencil Code
% CS 4476 / 6476: Computer Vision, Georgia Tech
% Written by Henry Hu

% Returns the camera center matrix for a given projection matrix

% 'Points_a' is nx2 matrix of 2D coordinate of points on Image A
% 'Points_b' is nx2 matrix of 2D coordinate of points on Image B
% 'F_matrix' is 3x3 fundamental matrix

% Try to implement this function as efficiently as possible. It will be
% called repeatly for part III of the project

function [ F ] = estimate_fundamental_matrix(Points_a,Points_b,normalize)

%%%%%%%%%%%%%%%%
% Your code here
%%%%%%%%%%%%%%%%

%This is an intentionally incorrect Fundamental matrix placeholder
% F_matrix = [0  0     -.0004; ...
%             0  0      .0032; ...
%             0 -0.0044 .1034];
num_pts = size(Points_a,1);

if normalize
    am=mean(Points_a);
    as=std(Points_a-am);
    Ta=[1/as(1) 0 0; 0 1/as(2) 0; 0 0 1]*[1 0 -am(1); 0 1 -am(2); 0 0 1];
    pts_a_nm = (Points_a-am)./as;

    bm=mean(Points_b);
    bs=std(Points_b-bm);
    Tb=[1/bs(1) 0 0; 0 1/bs(2) 0; 0 0 1]*[1 0 -bm(1); 0 1 -bm(2); 0 0 1];
    pts_b_nm = (Points_b-bm)./bs;
else
    pts_a_nm = Points_a;
    pts_b_nm = Points_b;
    Ta = eye(3);
    Tb = eye(3);
end

A = zeros(num_pts, 9);
for i=1:num_pts
    u1=pts_a_nm(i,1);
    v1=pts_a_nm(i,2);
    u2=pts_b_nm(i,1);
    v2=pts_b_nm(i,2);
    
    A(i,:)=[u1*u2 u2*v1 u2 v2*u1 v2*v1 v2 u1 v1 1];
end

%f = A\X;
[~, ~, V] = svd(A);
f = V(:, end);
F = reshape(f, [3 3])';
[U,S,V] = svd(F);
S(3,3)=0;
F=Tb'*U*S*V'*Ta;

end

