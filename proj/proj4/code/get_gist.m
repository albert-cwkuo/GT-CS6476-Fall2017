function image_feats = get_gist( image_paths )
N = numel(image_paths);


% Parameters for GIST feature
% Ref: http://people.csail.mit.edu/torralba/code/spatialenvelope/
clear param
param.imageSize = [256 256]; % it works also with non-square images
param.orientationsPerScale = [8 8 8 8];
param.numberBlocks = 4;
param.fc_prefilt = 4;

d = sum(param.orientationsPerScale)*param.numberBlocks^2;
image_feats = zeros(N, d);

% pre-compute gist
I = im2single(imread(image_paths{1}));
[gist, param] = LMgist(I, '', param);
image_feats(1,:) = gist/norm(gist);

parfor i=2:N
    I = im2single(imread(image_paths{i}));
    [gist, ~] = LMgist(I, '', param);
    image_feats(i,:) = gist/norm(gist);
end

end

