% RANSAC Stencil Code
% CS 4476 / 6476: Computer Vision, Georgia Tech
% Written by Henry Hu

% Find the best fundamental matrix using RANSAC on potentially matching
% points

% 'matches_a' and 'matches_b' are the Nx2 coordinates of the possibly
% matching points from pic_a and pic_b. Each row is a correspondence (e.g.
% row 42 of matches_a is a point that corresponds to row 42 of matches_b.

% 'Best_Fmatrix' is the 3x3 fundamental matrix
% 'inliers_a' and 'inliers_b' are the Mx2 corresponding points (some subset
% of 'matches_a' and 'matches_b') that are inliers with respect to
% Best_Fmatrix.

% For this section, use RANSAC to find the best fundamental matrix by
% randomly sample interest points. You would reuse
% estimate_fundamental_matrix() from part 2 of this assignment.

% If you are trying to produce an uncluttered visualization of epipolar
% lines, you may want to return no more than 30 points for either left or
% right images.

function [ Best_Fmatrix, inliers_a, inliers_b] = ransac_fundamental_matrix(matches_a, matches_b)


%%%%%%%%%%%%%%%%
% Your code here
%%%%%%%%%%%%%%%%

% Your ransac loop should contain a call to 'estimate_fundamental_matrix()'
% that you wrote for part II.

%placeholders, you can delete all of this
num_matches = size(matches_a,1);
e=0.7;
s=8;
p=0.99;
N=round(log(1-p)/log(1-(1-e)^s));
%N=70000; % assume that e=0.7, and we know that s=8, for p=0.99, we need N>=70188
eps=0.01;
supports = zeros(num_matches, N);
errs = zeros(num_matches, N);
Fs = zeros(3,3,N);
parfor n=1:N
    [samples_a,idx]=datasample(matches_a, 8, 'Replace', false);
    samples_b = matches_b(idx,:);
    Fe = estimate_fundamental_matrix(samples_a, samples_b, true);
    Fs(:,:,n) = Fe;
    for i=1:num_matches
        match_a = ones(1,3);
        match_a(1:2) = matches_a(i,:);
        match_b = ones(1,3);
        match_b = matches_b(i,:);
        match_b(end+1) = 1;
        v = abs(match_b*Fe*match_a');
        errs(i,n)=v;
        if v < eps
            supports(i,n)=1;
        end
    end
end

num_supports = zeros(1,N);
parfor n=1:N
    num_supports(n)=numel(find(supports(:,n)));
end
[max_v, match_ind] = max(num_supports);

matches_a(:, end+1) = errs(:, match_ind);
matches_b(:, end+1) = errs(:, match_ind);
inliers_a = sortrows(matches_a(supports(:, match_ind)>0,:),3);
inliers_b = sortrows(matches_b(supports(:, match_ind)>0,:),3);

Best_Fmatrix = Fs(:,:,match_ind);
inliers_a = inliers_a(1:40,1:2);
inliers_b = inliers_b(1:40,1:2);
end

