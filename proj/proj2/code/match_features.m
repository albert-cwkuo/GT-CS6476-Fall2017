% Local Feature Stencil Code
% CS 4476 / 6476: Computer Vision, Georgia Tech
% Written by James Hays

% 'features1' and 'features2' are the n x feature dimensionality features
%   from the two images.
% If you want to include geometric verification in this stage, you can add
% the x and y locations of the interest points as additional features.
%
% 'matches' is a k x 2 matrix, where k is the number of matches. The first
%   column is an index in features1, the second column is an index
%   in features2. 
% 'Confidences' is a k x 1 matrix with a real valued confidence for every
%   match.
% 'matches' and 'confidences' can empty, e.g. 0x2 and 0x1.
function [matches, confidences] = match_features(keypoints1, keypoints2)

% This function does not need to be symmetric (e.g. it can produce
% different numbers of matches depending on the order of the arguments).

% To start with, simply implement the "ratio test", equation 4.18 in
% section 4.1.3 of Szeliski. For extra credit you can implement various
% forms of spatial verification of matches.

% Placeholder that you can delete. Random matches and confidences
c_thres=0.6;
l1=numel(keypoints1);
l2=numel(keypoints2);


dist_mat = zeros(l1, l2);
parfor i=1:l1
    for j=1:l2
        dist_mat(i, j) = norm(keypoints1(i).feature-keypoints2(j).feature);
    end
end

matches = zeros(l1, 2);
confidences = zeros(l1, 1);
matches_count = 0;
for i=1:l1
    % match image 1 to image 2
    vd = dist_mat(i,:);
    [dmin1, ind1] = min(vd);
    vd(ind1) = 9999;
    [dmin2, ~] = min(vd);
    % match image 2 to image 1
    vd = dist_mat(:,ind1);
    vd(i)=9999;
    [dmin3, ~] = min(vd);
    if(dmin1/dmin2 <= 0.8 & dmin1/dmin3 <= 0.8)
    %if(dmin1/dmin2 <= c_thres)
        matches_count = matches_count+1;
        matches(matches_count,:)=[i, ind1];
        confidences(matches_count)=1/dmin1;
    end
end

matches = matches(1:matches_count, :);
confidences = confidences(1:matches_count, :);

% Sort the matches so that the most confident onces are at the top of the
% list. You should probably not delete this, so that the evaluation
% functions can be run on the top matches easily.
[confidences, ind] = sort(confidences, 'descend');
matches = matches(ind,:);