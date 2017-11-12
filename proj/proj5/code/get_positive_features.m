% Starter code prepared by James Hays for CS 4476, Georgia Tech
% This function should return all positive training examples (faces) from
% 36x36 images in 'train_path_pos'. Each face should be converted into a
% HoG template according to 'feature_params'. For improved performance, try
% mirroring or warping the positive training examples to augment your
% training data.

function features_pos = get_positive_features(train_path_pos, feature_params)
% 'train_path_pos' is a string. This directory contains 36x36 images of
%   faces
% 'feature_params' is a struct, with fields
%   feature_params.template_size (default 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.
%      (although you don't have to make the detector step size equal a
%      single HoG cell).


% 'features_pos' is N by D matrix where N is the number of faces and D
% is the template dimensionality, which would be
%   (feature_params.template_size / feature_params.hog_cell_size)^2 * 31
% if you're using the default vl_hog parameters

% Useful functions:
% vl_hog, HOG = VL_HOG(IM, CELLSIZE)
%  http://www.vlfeat.org/matlab/vl_hog.html  (API)
%  http://www.vlfeat.org/overview/hog.html   (Tutorial)
% rgb2gray

image_files = dir( fullfile( train_path_pos, '*.jpg') ); %Caltech Faces stored as .jpg
num_images = length(image_files);

% HoG feature
feature_dim = (feature_params.template_size / feature_params.hog_cell_size)^2 * 31;
features_hog = zeros(num_images, feature_dim);
for i=1:num_images
    % read and normalize image
    filename = fullfile(image_files(i).folder, image_files(i).name);
    I=im2single(imread(filename));
    if size(I,3) == 3
        I=rgb2gray(I);
    end
    Ir=imresize(I, [feature_params.template_size, feature_params.template_size]);
    % compute HoG feature
    HoG=vl_hog(Ir, feature_params.hog_cell_size);
    features_hog(i,:) = HoG(:)';
end

% GIST feature
% Parameters for GIST feature
% Ref: http://people.csail.mit.edu/torralba/code/spatialenvelope/
clear param
param.imageSize = [64 64]; % it works also with non-square images
param.orientationsPerScale = [8 8 8 8];
param.numberBlocks = 4;
param.fc_prefilt = 4;

feature_dim = sum(param.orientationsPerScale)*param.numberBlocks^2;
features_gist = zeros(num_images, feature_dim);

% pre-compute gist
% read and normalize image
filename = fullfile(image_files(1).folder, image_files(1).name);
I=im2double(imread(filename));
if size(I,3) == 3
    I=rgb2gray(I);
end
Ir=imresize(I, [64, 64]);
[gist, param] = LMgist(Ir, '', param);
features_gist(1,:) = gist;

parfor i=2:num_images
    filename = fullfile(image_files(i).folder, image_files(i).name);
    I=im2double(imread(filename));
    if size(I,3) == 3
        I=rgb2gray(I);
    end
    Ir=imresize(I, [64, 64]);
    [gist, ~] = LMgist(Ir, '', param);
    features_gist(i,:) = gist;
end

% concatenate HoG and GIST features
features_pos = horzcat(features_hog,features_gist);

end