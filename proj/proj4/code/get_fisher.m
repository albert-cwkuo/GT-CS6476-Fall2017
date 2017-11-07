function image_feats = get_fisher( image_paths )
% Ref: http://www.vlfeat.org/matlab/vl_fisher.html
load('gmm_mean.mat', 'gmm_mean');
load('gmm_covar.mat', 'gmm_covar');
load('gmm_prior.mat', 'gmm_prior');
gmm_mean = gmm_mean;
gmm_covar = gmm_covar;
gmm_prior = gmm_prior;


[d,k] = size(gmm_mean);

N = numel(image_paths);
image_feats = zeros(N, 2*d*k);
parfor i=1:N
    I = im2single(imread(image_paths{i}));
    [h,w] = size(I);
    step = floor(max([h,w])/50);
    [~, SIFT_features] = vl_dsift(I, 'Step', step, 'Fast');
    hist = vl_fisher(single(SIFT_features), gmm_mean, gmm_covar, gmm_prior);

    image_feats(i,:) = hist'/norm(hist);
end

end

