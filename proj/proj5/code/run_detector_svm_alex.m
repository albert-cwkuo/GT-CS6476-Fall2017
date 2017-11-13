% Starter code prepared by James Hays for CS 4476, Georgia Tech
% This function returns detections on all of the images in a given path.
% You will want to use non-maximum suppression on your detections or your
% performance will be poor (the evaluation counts a duplicate detection as
% wrong). The non-maximum suppression is done on a per-image basis. The
% starter code includes a call to a provided non-max suppression function.
function [bboxes, confidences, image_ids] = .... 
    run_detector_svm_alex(test_scn_path, w, b, feature_params)
% 'test_scn_path' is a string. This directory contains images which may or
%    may not have faces in them. This function should work for the MIT+CMU
%    test set but also for any other images (e.g. class photos)
% 'w' and 'b' are the linear classifier parameters
% 'feature_params' is a struct, with fields
%   feature_params.template_size (default 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.

% 'bboxes' is Nx4. N is the number of detections. bboxes(i,:) is
%   [x_min, y_min, x_max, y_max] for detection i. 
%   Remember 'y' is dimension 1 in Matlab!
% 'confidences' is Nx1. confidences(i) is the real valued confidence of
%   detection i.
% 'image_ids' is an Nx1 cell array. image_ids{i} is the image file name
%   for detection i. (not the full path, just 'albert.jpg')

% The placeholder version of this code will return random bounding boxes in
% each test image. It will even do non-maximum suppression on the random
% bounding boxes to give you an example of how to call the function.

% Your actual code should convert each test image to HoG feature space with
% a _single_ call to vl_hog for each scale. Then step over the HoG cells,
% taking groups of cells that are the same size as your learned template,
% and classifying them. If the classification is above some confidence,
% keep the detection and then pass all the detections for an image to
% non-maximum suppression. For your initial debugging, you can operate only
% at a single scale and you can skip calling non-maximum suppression. Err
% on the side of having a low confidence threshold (even less than zero) to
% achieve high enough recall.

% SVM confidence threshold
thres = -0.1;

% scales
scale_factor = 0.8;
final_scale = 0.2;
num_scales = ceil(log(final_scale)/log(scale_factor));

% alex net
net = alexnet;
reset(gpuDevice(1));

test_scenes = dir( fullfile( test_scn_path, '*.jpg' ));
%test_scenes.name='albert.jpg';
%test_scenes.folder='/home/albert/code/course/CV/proj/proj5/data/test_scenes/test_jpg';

%initialize these as empty and incrementally expand them.
num_proposals = (num_scales+1)*length(test_scenes);
bboxes = cell(num_proposals,1);
confidences = cell(num_proposals,1);
image_ids = cell(num_proposals,1);

%for i = 1:3
for i = 1:length(test_scenes)
    fprintf('Detecting faces in %s\n', test_scenes(i).name)
    img = imread( fullfile( test_scn_path, test_scenes(i).name ));
    img = im2single(img);
    if(size(img,3) > 1)
        img = rgb2gray(img);
    end
    [img_h, img_w] = size(img);
    
    % loop over different scales
    bboxes_scale = cell(num_scales+1,1);
    confidences_scale = cell(num_scales+1,1);
    image_ids_scale = cell(num_scales+1,1);
    for j = 1:num_scales+1
        scale = scale_factor^(j-1);
        img_s = imresize(img,scale);
        % compute HoG ove the whole image
        HoG=vl_hog(img_s, feature_params.hog_cell_size);
        height=size(HoG,1);
        width=size(HoG,2);
        d=feature_params.template_size/feature_params.hog_cell_size;
        fd=d^2*31;
        cell_w=width-d+1;
        cell_h=height-d+1;
        if or(cell_w<=0,cell_h<=0)
            num_cell=0;
        else
            num_cell=cell_w*cell_h;
        end
        HoG_cell = zeros(num_cell, fd);
        cur_bboxes = zeros(num_cell, 4);
        cur_image_ids = cell(num_cell,1);
        for x=1:cell_w
            for y=1:cell_h
                % HoG cell feature
                HoG_cell_xy = HoG(y:y+d-1,x:x+d-1,:);
                ind=y+(x-1)*cell_h;
                HoG_cell(ind,:)=HoG_cell_xy(:)';
                % cell bbox
                xmin=floor((x-1)*feature_params.hog_cell_size/scale)+1;
                xmax=min(img_w, xmin+floor(feature_params.template_size/scale)-1);
                ymin=floor((y-1)*feature_params.hog_cell_size/scale)+1;
                ymax=min(img_h, ymin+floor(feature_params.template_size/scale)-1);
                cur_bboxes(ind,:)=[xmin,ymin,xmax,ymax];
                % image id
                cur_image_ids{ind}=test_scenes(i).name;
            end
        end
        % alexnet feature
        % split it into serveral pieces to avoid out of memory problem
        split_size = 1024;
        if num_cell > split_size
            features_alex = zeros(num_cell, 4096);
            % major parts
            for k=1:split_size:num_cell-split_size
                cur_images = zeros(227,227,3,split_size);
                for l=k:k+split_size-1
                    xmin=cur_bboxes(l,1);
                    ymin=cur_bboxes(l,2);
                    xmax=cur_bboxes(l,3);
                    ymax=cur_bboxes(l,4);
                    It=img(ymin:ymax,xmin:xmax,:);
                    It=imresize(It,[227,227]);
                    if size(It,3) ~= 3
                        It=cat(3,It,It,It);
                    end
                    cur_images(:,:,:,l-k+1)=It;
                end
                alex=activations(net,cur_images,'fc7');
                features_alex(k:k+split_size-1,:)=alex;
            end
            % remaining parts
            rem_size = size(cur_bboxes(k+split_size:end,:),1);
            cur_images = zeros(227,227,3,rem_size);
            for l=k+split_size:size(cur_bboxes,1)
                xmin=cur_bboxes(l,1);
                ymin=cur_bboxes(l,2);
                xmax=cur_bboxes(l,3);
                ymax=cur_bboxes(l,4);
                It=img(ymin:ymax,xmin:xmax,:);
                It=imresize(It,[227,227]);
                if size(It,3) ~= 3
                    It=cat(3,It,It,It);
                end
                cur_images(:,:,:,l-k-split_size+1)=It;
            end
            alex=activations(net,cur_images,'fc7');
            features_alex(k+split_size:end,:)=alex;
        elseif num_cell > 0
            cur_images = zeros(227,227,3,num_cell);
            for l=1:num_cell
                xmin=cur_bboxes(l,1);
                ymin=cur_bboxes(l,2);
                xmax=cur_bboxes(l,3);
                ymax=cur_bboxes(l,4);
                It=img(ymin:ymax,xmin:xmax,:);
                It=imresize(It,[227,227]);
                if size(It,3) ~= 3
                    It=cat(3,It,It,It);
                end
                cur_images(:,:,:,l)=It;
            end
            alex=activations(net,cur_images,'fc7');
            features_alex=alex;
        else % num_cell <= 0
            features_alex=zeros(0, 4096);
        end
        % confidence for each sliding window
        features=horzcat(HoG_cell, double(features_alex));
        cur_confidences=features*w+b;
        % filter out those confidence < thres
        hit = cur_confidences > thres;
        cur_bboxes = cur_bboxes(hit,:);
        cur_confidences = cur_confidences(hit,:);
        cur_image_ids = cur_image_ids(hit,:);
        % collect all results in this scale
        bboxes_scale{j} = cur_bboxes;
        confidences_scale{j} = cur_confidences;
        image_ids_scale{j} = cur_image_ids;
    end
    %non_max_supr_bbox can actually get somewhat slow with thousands of
    %initial detections. You could pre-filter the detections by confidence,
    %e.g. a detection with confidence -1.1 will probably never be
    %meaningful. You probably _don't_ want to threshold at 0.0, though. You
    %can get higher recall with a lower threshold. You don't need to modify
    %anything in non_max_supr_bbox, but you can.
    bboxes_scale = cell2mat(bboxes_scale);
    confidences_scale = cell2mat(confidences_scale);
    image_ids_scale = vertcat(image_ids_scale{:});
    
    [is_maximum] = non_max_supr_bbox(bboxes_scale, confidences_scale, size(img));
    confidences_scale = confidences_scale(is_maximum,:);
    bboxes_scale = bboxes_scale(is_maximum,:);
    image_ids_scale = image_ids_scale(is_maximum,:);
    % collect all results in this image
    bboxes{i} = bboxes_scale;
    confidences{i} = confidences_scale;
    image_ids{i} = image_ids_scale;
end
bboxes = cell2mat(bboxes);
confidences = cell2mat(confidences);
image_ids = vertcat(image_ids{:});
end




