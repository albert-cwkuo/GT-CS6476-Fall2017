% Starter code prepared by James Hays for Computer Vision

%This function will sample SIFT descriptors from the training images,
%cluster them with kmeans, and then return the cluster centers.

function vocab = build_vocabulary( image_paths, vocab_size, spatial_pyramid )
% The inputs are 'image_paths', a N x 1 cell array of image paths, and
% 'vocab_size' the size of the vocabulary.

% The output 'vocab' should be vocab_size x 128. Each row is a cluster
% centroid / visual word.

%{
Useful functions:
[locations, SIFT_features] = vl_dsift(img) 
 http://www.vlfeat.org/matlab/vl_dsift.html
 locations is a 2 x n list list of locations, which can be thrown away here
  (but possibly used for extra credit in get_bags_of_sifts if you're making
  a "spatial pyramid").
 SIFT_features is a 128 x N matrix of SIFT features
  note: there are step, bin size, and smoothing parameters you can
  manipulate for vl_dsift(). We recommend debugging with the 'fast'
  parameter. This approximate version of SIFT is about 20 times faster to
  compute. Also, be sure not to use the default value of step size. It will
  be very slow and you'll see relatively little performance gain from
  extremely dense sampling. You are welcome to use your own SIFT feature
  code! It will probably be slower, though.

[centers, assignments] = vl_kmeans(X, K)
 http://www.vlfeat.org/matlab/vl_kmeans.html
  X is a d x M matrix of sampled SIFT features, where M is the number of
   features sampled. M should be pretty large! Make sure matrix is of type
   single to be safe. E.g. single(matrix).
  K is the number of clusters desired (vocab_size)
  centers is a d x K matrix of cluster centroids. This is your vocabulary.
   You can disregard 'assignments'.

  Matlab has a built in kmeans function, see 'help kmeans', but it is
  slower.
%}

% Load images from the training set. To save computation time, you don't
% necessarily need to sample from all images, although it would be better
% to do so. You can randomly sample the descriptors from each image to save
% memory and speed up the clustering. Or you can simply call vl_dsift with
% a large step size here, but a smaller step size in get_bags_of_sifts.m. 

% For each loaded image, get some SIFT features. You don't have to get as
% many SIFT features as you will in get_bags_of_sift.m, because you're only
% trying to get a representative sample here.

% Once you have tens of thousands of SIFT features from many training
% images, cluster them with kmeans. The resulting centroids are now your
% visual word vocabulary.

if spatial_pyramid
    N = numel(image_paths);
    pyr = 2;
    div = 2^(pyr-1);
    F = cell(div, div, N);
    parfor n=1:N
        I = im2single(imread(image_paths{n}));
        [h,w] = size(I);
        step = floor(max([h,w])/20);
        for x=1:div
            for y=1:div
                xmin=1+(x-1)*floor(w/div);
                xmax=x*floor(w/div);
                ymin=1+(y-1)*floor(h/div);
                ymax=y*floor(h/div);
                Ipatch = I(ymin:ymax,xmin:xmax);
                [~, f] = vl_dsift(Ipatch, 'Step', step, 'Fast');
                F{x,y,n}=f';
            end
        end
    end

    vocab=cell(pyr,1);
    parfor p=1:pyr
        d = 2^(p-1);
        vocab_p=zeros(d,d,vocab_size, 128);
        for x=1:d
            for y=1:d
                xmin=1+(x-1)*div/d;
                xmax=x*div/d;
                ymin=1+(y-1)*div/d;
                ymax=y*div/d;
                f=F(xmin:xmax, ymin:ymax, :);
                f=cell2mat(reshape(f, [], 1));
                [c, ~] = vl_kmeans(single(f'), vocab_size);
                vocab_p(x,y,:,:)=c';
            end
        end
        vocab{p}=vocab_p;
    end
    a=1;
else
    N = numel(image_paths);
    F = cell(N,1);
    parfor i=1:N
        I = im2single(imread(image_paths{i}));
        [h,w] = size(I);
        step = floor(max([h,w])/20);

        [~, SIFT_features] = vl_dsift(I, 'Step', step, 'Fast');
        F{i} = SIFT_features';
    end

    Fmat = cell2mat(F);

    [centers, ~] = vl_kmeans(single(Fmat'), vocab_size);
    vocab = centers';
end
end