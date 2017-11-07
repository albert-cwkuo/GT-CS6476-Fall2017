function [mean, covar, prior] = build_gmm( image_paths, vocab_size )
% Ref: http://www.vlfeat.org/matlab/vl_gmm.html
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

[mean, covar, prior] = vl_gmm(single(Fmat'), vocab_size);

end