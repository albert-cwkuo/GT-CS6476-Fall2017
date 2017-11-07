function visualizeKeypoints(image, keypoints, name)
numkp = numel(keypoints);
color = rand(numkp, 3);
% draw circle
cx = [keypoints.x_abs]';
cy = [keypoints.y_abs]';
r = [keypoints.sigma]';
image = insertShape(image,'Circle',[cx, cy, r],'LineWidth',2,'Color',color);
% draw orientation
theta = [keypoints.ori]'/180*pi;
ex = cx+r.*cos(theta);
ey = cy+r.*sin(theta);
image = insertShape(image,'Line',[cx cy ex ey],'LineWidth',2,'Color',color);
% show image
figure
imshow(image)
imwrite(image, name, 'quality', 100);
end