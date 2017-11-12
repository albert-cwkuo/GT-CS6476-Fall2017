function I = random_sample_negative_patches(image_files, dim)
% randomly sample one image
num_images = length(image_files);
ind = randi(num_images);
filename = fullfile(image_files(ind).folder, image_files(ind).name);
I=imread(filename);
% randomly scale image
scale_min = 0.4;
if scale_min*min(size(I,1), size(I,2)) > dim
    scale = scale_min+(1-scale_min)*rand();
    I=imresize(I,scale);
end
% randomly crop image
h=size(I,1);
w=size(I,2);
xmin=randi(w-dim+1);
ymin=randi(h-dim+1);
I=imcrop(I,[xmin,ymin,dim-1,dim-1]);

end

