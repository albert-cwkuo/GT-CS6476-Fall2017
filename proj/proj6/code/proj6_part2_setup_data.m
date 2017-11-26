function imdb = proj6_part2_setup_data(averageImage)
%code for Computer Vision, Georgia Tech by James Hays

% It's probably easiest if you start by copy/paste the contents of
% proj6_part1_setup_data, including the normalization by removing the mean.

% [copied from the project webpage]
% The input images need to be resized to 224x224. More specifically, the
% input images need to be 224x224 when returned by getBatch(). You could
% keep them at higher resolution in imdb and crop them to 224x224 as a form
% of jittering. See cnn_imagenet_get_batch.m for an extreme jittering
% example. You can call that function if you want, but you can achieve high
% accuracy with no jittering. You could also simply reuse your jittering
% strategy from Part 1.
   
% VGG-F accepts 3 channel (RGB) images. The 15 scene database contains
% grayscale images. There are two possibilities: modify the first layer of
% VGG-F to accept 1 channel images, or concatenate the grayscale images
% with themselves (e.g. cat(3, im, im, im)) to make an RGB image. The
% latter is probably easier and safer.
  
% VGG-F expects input images to be normalized by subtracting the average
% image, just like in Part 1. VGG-F provides a 224x224x3 average image in
% net.meta.normalization.averageImage.
