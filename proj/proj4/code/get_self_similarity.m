function image_feats = get_self_similarity( image_paths )
clear parms
parms.patch_size=5;
parms.desc_rad=40;
parms.nrad=3;
parms.nang=12;
parms.var_noise=300000;
parms.saliency_thresh=0.7;
parms.homogeneity_thresh=0.7;
parms.snn_thresh=0.85;

img = im2double(imread(image_paths{1}));
[resp,drawCoords,salientCoords,homogeneousCoords,snnCoords]=mexCalcSsdescs(img,parms);

end

