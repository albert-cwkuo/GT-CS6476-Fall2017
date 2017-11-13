% Starter code prepared by James Hays for CS 4476, Georgia Tech
% This function should return all positive training examples (faces) from
% 36x36 images in 'train_path_pos'. Each face should be converted into a
% HoG template according to 'feature_params'. For improved performance, try
% mirroring or warping the positive training examples to augment your
% training data.

function features_pos = get_positive_features(train_path_pos, feature_params, alexnet)
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

feature_dim = (feature_params.template_size / feature_params.hog_cell_size)^2 * 31;
features_hog = zeros(num_images, feature_dim);
images = zeros(227,227,3,num_images);
parfor i=1:num_images
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
    % collect images for alexnet
    Ir=imresize(Ir,[227,227]);
    Ir=cat(3,Ir,Ir,Ir);
    images(:,:,:,i)=Ir;
end

if alexnet
% compute AlexNet feature
net = alexnet;
reset(gpuDevice(1));
% split it into serveral pieces to avoid out of memory problem
split_size = 1024;
features_alex = zeros(num_images, 4096);
for i=1:split_size:num_images-split_size
    alex=activations(net,images(:,:,:,i:i+split_size-1),'fc7');
    features_alex(i:i+split_size-1,:)=alex;
end
alex=activations(net,images(:,:,:,i+split_size:end),'fc7');
features_alex(i+split_size:end,:)=alex;

features_pos = horzcat(features_hog, features_alex);
else
    features_pos =features_hog;
end

end