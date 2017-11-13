% Starter code prepared by James Hays for CS 4476, Georgia Tech
% This function should return negative training examples (non-faces) from
% any images in 'non_face_scn_path'. Images should be converted to
% grayscale because the positive training data is only available in
% grayscale. For best performance, you should sample random negative
% examples at multiple scales.

function features_neg = get_random_negative_features(non_face_scn_path, feature_params, num_samples, alexnet)
% 'non_face_scn_path' is a string. This directory contains many images
%   which have no faces in them.
% 'feature_params' is a struct, with fields
%   feature_params.template_size (default 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.
% 'num_samples' is the number of random negatives to be mined, it's not
%   important for the function to find exactly 'num_samples' non-face
%   features, e.g. you might try to sample some number from each image, but
%   some images might be too small to find enough.

% 'features_neg' is N by D matrix where N is the number of non-faces and D
% is the template dimensionality, which would be
%   (feature_params.template_size / feature_params.hog_cell_size)^2 * 31
% if you're using the default vl_hog parameters

% Useful functions:
% vl_hog, HOG = VL_HOG(IM, CELLSIZE)
%  http://www.vlfeat.org/matlab/vl_hog.html  (API)
%  http://www.vlfeat.org/overview/hog.html   (Tutorial)
% rgb2gray

image_files = dir( fullfile( non_face_scn_path, '*.jpg' ));

feature_dim = (feature_params.template_size / feature_params.hog_cell_size)^2 * 31;
features_hog = zeros(num_samples, feature_dim);
images = zeros(227,227,3,num_samples);
parfor i=1:num_samples
    % randomly sample negative patch image
    I = random_sample_negative_patches(image_files, feature_params.template_size);
    % normalize image
    I=im2single(I);
    if size(I,3) == 3
        I=rgb2gray(I);
    end
    % compute HoG feature
    HoG=vl_hog(I, feature_params.hog_cell_size);
    features_hog(i,:) = HoG(:)';
    % collect images for alexnet
    Ir=imresize(I,[227,227]);
    Ir=cat(3,Ir,Ir,Ir);
    images(:,:,:,i)=Ir;
end

if alexnet
% compute AlexNet feature
net = alexnet;
reset(gpuDevice(1));
% split it into serveral pieces to avoid out of memory problem
split_size = 1024;
features_alex = zeros(num_samples, 4096);
for i=1:split_size:num_samples-split_size
    alex=activations(net,images(:,:,:,i:i+split_size-1),'fc7');
    features_alex(i:i+split_size-1,:)=alex;
end
alex=activations(net,images(:,:,:,i+split_size:end),'fc7');
features_alex(i+split_size:end,:)=alex;

features_neg = horzcat(features_hog, features_alex);
else
    features_neg = features_hog;
end

end