function I = random_sample_negative_patches(image_files, dim)
num_images = length(image_files);
% randomly sample one image
ind = randi(num_images);
filename = fullfile(image_files(ind).folder, image_files(ind).name);
I=imread(filename);
% randomly crop image
h=size(I,1);
w=size(I,2);
xmin=randi(w-dim+1);
ymin=randi(h-dim+1);
I=imcrop(I,[xmin,ymin,dim-1,dim-1]);

end

