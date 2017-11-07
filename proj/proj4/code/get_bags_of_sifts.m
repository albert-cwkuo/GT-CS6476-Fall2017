% Starter code prepared by James Hays for Computer Vision

%This feature representation is described in the handout, lecture
%materials, and Szeliski chapter 14.

function image_feats = get_bags_of_sifts(image_paths, soft_assignment, spatial_pyramid)
% image_paths is an N x 1 cell array of strings where each string is an
% image path on the file system.

% This function assumes that 'vocab.mat' exists and contains an N x 128
% matrix 'vocab' where each row is a kmeans centroid or visual word. This
% matrix is saved to disk rather than passed in a parameter to avoid
% recomputing the vocabulary every run.

% image_feats is an N x d matrix, where d is the dimensionality of the
% feature representation. In this case, d will equal the number of clusters
% or equivalently the number of entries in each image's histogram
% ('vocab_size') below.

% You will want to construct SIFT features here in the same way you
% did in build_vocabulary.m (except for possibly changing the sampling
% rate) and then assign each local feature to its nearest cluster center
% and build a histogram indicating how many times each cluster was used.
% Don't forget to normalize the histogram, or else a larger image with more
% SIFT features will look very different from a smaller version of the same
% image.

%{
Useful functions:
[locations, SIFT_features] = vl_dsift(img) 
 http://www.vlfeat.org/matlab/vl_dsift.html
 locations is a 2 x n list list of locations, which can be used for extra
  credit if you are constructing a "spatial pyramid".
 SIFT_features is a 128 x N matrix of SIFT features
  note: there are step, bin size, and smoothing parameters you can
  manipulate for vl_dsift(). We recommend debugging with the 'fast'
  parameter. This approximate version of SIFT is about 20 times faster to
  compute. Also, be sure not to use the default value of step size. It will
  be very slow and you'll see relatively little performance gain from
  extremely dense sampling. You are welcome to use your own SIFT feature
  code! It will probably be slower, though.

D = vl_alldist2(X,Y) 
   http://www.vlfeat.org/matlab/vl_alldist2.html
    returns the pairwise distance matrix D of the columns of X and Y. 
    D(i,j) = sum (X(:,i) - Y(:,j)).^2
    Note that vl_feat represents points as columns vs this code (and Matlab
    in general) represents points as rows. So you probably want to use the
    transpose operator '  You can use this to figure out the closest
    cluster center for every SIFT feature. You could easily code this
    yourself, but vl_alldist2 tends to be much faster.

Or:

For speed, you might want to play with a KD-tree algorithm (we found it
reduced computation time modestly.) vl_feat includes functions for building
and using KD-trees.
 http://www.vlfeat.org/matlab/vl_kdtreebuild.html

%}

if spatial_pyramid
    load('vocab_sp.mat', 'vocab');
    vocab = vocab;
    pyr = numel(vocab);
    vocab_size = size(vocab{1}, 3);
    r=100;

    N = numel(image_paths);
    div = 2^(pyr-1);
    image_feats = cell(N, 1);
    parfor n=1:N
        I = im2single(imread(image_paths{n}));
        [h,w] = size(I);
        step = floor(max([h,w])/50);
        F = cell(div, div);
        for x=1:div
            for y=1:div
                xmin=1+(x-1)*floor(w/div);
                xmax=x*floor(w/div);
                ymin=1+(y-1)*floor(h/div);
                ymax=y*floor(h/div);
                Ipatch = I(ymin:ymax,xmin:xmax);
                [~, f] = vl_dsift(Ipatch, 'Step', step, 'Fast');
                F{x,y}=f';
            end
        end

        hist = cell(1,pyr);
        for p=1:pyr
            d = 2^(p-1);
            hist_p = zeros(1,d*d*vocab_size);
            for x=1:d
                for y=1:d
                    xmin=1+(x-1)*div/d;
                    xmax=x*div/d;
                    ymin=1+(y-1)*div/d;
                    ymax=y*div/d;
                    f=F(xmin:xmax, ymin:ymax);
                    f=cell2mat(reshape(f, [], 1));
                    D = vl_alldist2(squeeze(vocab{p}(x,y,:,:))', double(f'), 'CHI2');
                    if soft_assignment
                        for col=1:size(D,2)
                            D(:,col)=D(:,col)/norm(D(:,col));
                        end
                        hist_xy = sum(softmax(-D*r), 2);
                    else
                        [~, ind] = min(D);
                        [hist_xy,~] = histcounts(ind, 'BinLimits', [1-0.5, vocab_size+0.5], 'BinMethod', 'integers');
                    end

                    hmin=1+(x-1)*d*vocab_size + (y-1)*vocab_size;
                    hmax=(x-1)*d*vocab_size + y*vocab_size;
                    hist_p(hmin:hmax) = hist_xy';
                end
            end
            hist{p}=hist_p/norm(hist_p);
        end
        hist = cell2mat(hist);
        image_feats{n} = hist/norm(hist);
    end
    image_feats = cell2mat(image_feats);
    a=1;
else
    load('vocab.mat', 'vocab');
    vocab_size = size(vocab, 1);
    vocab = vocab';
    
    r=100;

    N = numel(image_paths);
    image_feats = zeros(N, vocab_size);
    parfor i=1:N
        I = im2single(imread(image_paths{i}));
        [h,w] = size(I);
        step = floor(max([h,w])/50);

        [~, SIFT_features] = vl_dsift(I, 'Step', step, 'Fast');
        D = vl_alldist2(vocab, single(SIFT_features));
        if soft_assignment
            for col=1:size(D,2)
                D(:,col)=D(:,col)/norm(D(:,col));
            end
            hist = sum(softmax(-D*r), 2);
        else
            [~, ind] = min(D);
            [hist,edges] = histcounts(ind, 'BinLimits', [1-0.5, vocab_size+0.5], 'BinMethod', 'integers');  
        end
        
        image_feats(i,:) = hist'/norm(hist);
    end
end

end