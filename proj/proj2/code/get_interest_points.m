% Local Feature Stencil Code
% CS 4476 / 6476: Computer Vision, Georgia Tech
% Written by James Hays

% Returns a set of interest points for the input image

% 'image' can be grayscale or color, your choice.
% 'feature_width', in pixels, is the local feature width. It might be
%   useful in this function in order to (a) suppress boundary interest
%   points (where a feature wouldn't fit entirely in the image, anyway)
%   or (b) scale the image filters being used. Or you can ignore it.

% 'x' and 'y' are nx1 vectors of x and y coordinates of interest points.
% 'confidence' is an nx1 vector indicating the strength of the interest
%   point. You might use this later or not.
% 'scale' and 'orientation' are nx1 vectors indicating the scale and
%   orientation of each interest point. These are OPTIONAL. By default you
%   do not need to make scale and orientation invariant local features.
function [keypoints, DGS] = get_interest_points(image)

% Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
% You can create additional interest point detector functions (e.g. MSER)
% for extra credit.

% If you're finding spurious interest point detections near the boundaries,
% it is safe to simply suppress the gradients / corners near the edges of
% the image.

% The lecture slides and textbook are a bit vague on how to do the
% non-maximum suppression once you've thresholded the cornerness score.
% You are free to experiment. Here are some helpful functions:
%  BWLABEL and the newer BWCONNCOMP will find connected components in 
% thresholded binary image. You could, for instance, take the maximum value
% within each component.
%  COLFILT can be used to run a max() operator on each sliding window. You
% could use this to ensure that every interest point is at a local maximum
% of cornerness.

% default parameters
global n_oct;
global n_spo;
global sig_min;
global sig_in;
global del_min;
global starting_oct;
n_oct=5;
n_spo=3;
sig_min=0.8;
sig_in=0.5;
del_min=0.5;
starting_oct=2;

% compute discrete gaussian space
DGS=contructDGS(image);
% compute difference of gaussian
DoG=computeDoG(DGS);
% get sift keypoint
keypoints1=getSIFTKeypoints(DoG);
% get harris keypoints
%[keypoints2, DGS]=getHarrisKeypoints(DGS);
% get keypoint orientation
%keypoints = [keypoints1, keypoints2];
keypoints = keypoints1;
[keypoints, DGS]=computeOrientation(keypoints, DGS);
% for ablation test only
%keypoints=removeScaleInvariance(keypoints);
%keypoints=removeRotationInvariance(keypoints);
end

function DGS=contructDGS(image)
% default parameters
global n_oct;
global n_spo;
global sig_min;
global sig_in;
global del_min;

% Discrete Gaussian Space
% compute the starting image
% note the shift +1 of scale index
% u0 = imresize(image, 1/del_min, 'nearest');
u0 = imresize(image, 1/del_min);
gsig = sqrt(sig_min^2-sig_in^2)/del_min;
v0 = imgaussfilt(u0, gsig);
del0 = del_min;
sig = sig_min;
scale(1).image=v0;
scale(1).sigma=sig;
octave(1).scale = scale;
octave(1).delta = del0;
DGS.octave=octave;

% compute first octave
for sc=1:n_spo+2
    gsig = sig_min/del_min*sqrt(2^(2*sc/n_spo)-2^(2*(sc-1)/n_spo));
    v = imgaussfilt(DGS.octave(1).scale(sc).image, gsig);
    sig = del0/del_min*sig_min*2^(sc/n_spo);
    DGS.octave(1).scale(sc+1).image = v;
    DGS.octave(1).scale(sc+1).sigma = sig;
end

% compute rest octaves
for oct=2:n_oct
    del_oct = del_min*2^(oct-1);
    v0 = DGS.octave(oct-1).scale(4).image(1:2:end, 1:2:end);
    sig = del_oct/del_min*sig_min;
    DGS.octave(oct).scale(1).image = v0;
    DGS.octave(oct).scale(1).sigma = sig;
    DGS.octave(oct).delta = del_oct;
    for sc=1:n_spo+2
        gsig = sig_min/del_min*sqrt(2^(2*sc/n_spo)-2^(2*(sc-1)/n_spo));
        v = imgaussfilt(DGS.octave(oct).scale(sc).image, gsig);
        sig = del_oct/del_min*sig_min*2^(sc/n_spo);
        DGS.octave(oct).scale(sc+1).image = v;
        DGS.octave(oct).scale(sc+1).sigma = sig;
    end
end
end

function DoG=computeDoG(DGS)
n_oct = size(DGS.octave, 2);
n_spo = size(DGS.octave(1).scale, 2);
DoG=DGS;
for oct=1:n_oct
    for sc=1:n_spo-1
        DoG.octave(oct).scale(sc).image...
            = DoG.octave(oct).scale(sc+1).image...
            - DoG.octave(oct).scale(sc).image;
    end
    DoG.octave(oct).scale = DoG.octave(oct).scale(1:n_spo-1);
end
end

function keypoints=getSIFTKeypoints(DoG)
C_dog = 0.015;
C_edge = 10;
C_edgeness = (C_edge+1)^2/C_edge;
global sig_min;
global del_min;
global n_oct;
global n_spo;
global starting_oct;

% pre-allocate the keypoints array
l = floor(numel(DoG.octave(2).scale(1).image)/20);
keypoints(l).x_dgs = 0;
num_keypoints=0;
for oct=starting_oct:n_oct
    [h, w] = size(DoG.octave(oct).scale(1).image);
    for sc=2:n_spo+1
        for y=2:h-1
            for x=2:w-1
                % check contrast
                val = DoG.octave(oct).scale(sc).image(y, x);
                if abs(val) < 0.8*C_dog
                    continue
                end
                % check local extremum
                neighbor = [DoG.octave(oct).scale(sc-1).image(y-1:y+1, x-1:x+1),...
                            DoG.octave(oct).scale(sc).image(y-1:y+1, x-1:x+1),...
                            DoG.octave(oct).scale(sc+1).image(y-1:y+1, x-1:x+1)];
                if any(abs(val) < abs(neighbor(:)))
                    continue;
                end
                % check edge point
                patch = DoG.octave(oct).scale(sc).image(y-1:y+1, x-1:x+1);
                h11 = patch(2,3)+patch(2,1)-2*patch(2,2);
                h22 = patch(3,2)+patch(1,2)-2*patch(2,2);
                h12 = (patch(1,1)+patch(3,3)-patch(1,3)-patch(3,1))/4;
                hess = [h11, h12; h12, h22];
                edgeness = (trace(hess))^2/det(hess);
                if edgeness > C_edgeness
                    continue;
                end
                % interpolate keypoint
                [alpha, xi, yi, si, val] = interpolateSIFTKeypoint(DoG,x,y,oct,sc);
                if max(abs(alpha(:))) >= 0.6 | abs(val) < C_dog
                    continue;
                end
                % record keypoint
                del = DoG.octave(oct).delta;
                s_dgs = si+alpha(1);
                x_dgs = xi+alpha(2);
                y_dgs = yi+alpha(3);
                sig = del/del_min*sig_min*2^((si+alpha(1)-1)/n_spo);
                x_abs = 1+del*(xi+alpha(2)-1);
                y_abs = 1+del*(yi+alpha(3)-1);
                num_keypoints = num_keypoints+1;
                keypoints(num_keypoints).x_dgs=x_dgs;
                keypoints(num_keypoints).y_dgs=y_dgs;
                keypoints(num_keypoints).s_dgs=s_dgs;
                keypoints(num_keypoints).octave=oct;
                keypoints(num_keypoints).x_abs=x_abs;
                keypoints(num_keypoints).y_abs=y_abs;
                keypoints(num_keypoints).sigma=sig;
            end
        end
    end
end
keypoints = keypoints(1:num_keypoints);
end

function [keypoints, DGS]=getHarrisKeypoints(DGS)
global n_oct;
global n_spo;
global del_min;
global sig_min;
global starting_oct;
evect_alpha = 0.05;
cornerness=0.001;

% compute image gradient
for o=starting_oct:n_oct
    for s=1:n_spo+2
        [Gx,Gy] = imgradientxy(DGS.octave(o).scale(s).image);
        [gmag,gdir] = imgradient(Gx,Gy);% dir: -180~180
        DGS.octave(o).scale(s).gmag=gmag;
        DGS.octave(o).scale(s).gdir=gdir+180; % dir: 0~360
        Gxx = Gx .* Gx;
        Gyy = Gy .* Gy;
        Gxy = Gx .* Gy;
        edge_window = DGS.octave(o).scale(s).sigma/DGS.octave(o).delta;
        gGxx = imgaussfilt(Gxx,edge_window);
        gGyy = imgaussfilt(Gyy,edge_window);
        gGxy = imgaussfilt(Gxy,edge_window);
        R = gGxx .* gGyy - gGxy .* gGxy - evect_alpha * (gGxx + gGyy).^2;
        DGS.octave(o).scale(s).R=R;
    end
end


% pre-allocate the keypoints array
l = floor(numel(DGS.octave(2).scale(1).image)/20);
keypoints(l).x_dgs = 0;
num_keypoints=0;
for o=2:n_oct
    [h, w] = size(DGS.octave(o).scale(1).R);
    for s=1:n_spo+2
        for y=2:h-1
            for x=2:w-1
                % check cornerness
                val = DGS.octave(o).scale(s).R(y, x);
                if val < 0.8*cornerness
                    continue
                end
                % check local extremum
                neighbor = DGS.octave(o).scale(s).R(y-1:y+1, x-1:x+1);
                if any(val < neighbor(:) | val < 0)
                    continue;
                end
                % interpolate keypoint
                [alpha, xi, yi, vali] = interpolateHarrisKeypoint(DGS,x,y,o,s);
                if max(alpha(:)) >= 0.6 | abs(vali) < cornerness
                    continue;
                end
                % record keypoint
                del = DGS.octave(o).delta;
                s_dgs = s;
                x_dgs = xi+alpha(1);
                y_dgs = yi+alpha(2);
                sig = del/del_min*sig_min*2^((s-1)/n_spo);
                x_abs = 1+del*(xi+alpha(1)-1);
                y_abs = 1+del*(yi+alpha(2)-1);
                num_keypoints = num_keypoints+1;
                keypoints(num_keypoints).x_dgs=x_dgs;
                keypoints(num_keypoints).y_dgs=y_dgs;
                keypoints(num_keypoints).s_dgs=s_dgs;
                keypoints(num_keypoints).octave=o;
                keypoints(num_keypoints).x_abs=x_abs;
                keypoints(num_keypoints).y_abs=y_abs;
                keypoints(num_keypoints).sigma=sig;
            end
        end
    end
end
keypoints = keypoints(1:num_keypoints);
end

function [alpha, x, y, s, val] = interpolateSIFTKeypoint(DoG,x,y,o,s)
global n_spo;
[h,w] = size(DoG.octave(o).scale(1).image);
for i=1:5
    patch = cat(3, DoG.octave(o).scale(s-1).image(y-1:y+1, x-1:x+1),...
               DoG.octave(o).scale(s).image(y-1:y+1, x-1:x+1),...
               DoG.octave(o).scale(s+1).image(y-1:y+1, x-1:x+1));
    [alpha, hess, grad] = quadraticInterpolate3D(patch);
    if max(abs(alpha(:))) < 0.6
        break;
    else
        alpha(alpha>=0.6) = 0.6;
        alpha(alpha<=-0.6) = -0.6;
        s=min(max(round(s+alpha(1)), 2), n_spo+1);
        x=min(max(round(x+alpha(2)), 2), w-1);
        y=min(max(round(y+alpha(3)), 2), h-1);
    end
end
val = patch(2,2,2)-0.5*transpose(grad)*(hess\grad);
end

function [alpha, x, y, val] = interpolateHarrisKeypoint(DGS,x,y,o,s)
global n_spo;
[h,w] = size(DGS.octave(o).scale(1).R);
for i=1:5
    patch = DGS.octave(o).scale(s).R(y-1:y+1, x-1:x+1);
    [alpha, hess, grad] = quadraticInterpolate2D(patch);
    if max(abs(alpha(:))) < 0.6
        break;
    else
        alpha(alpha>=0.6) = 0.6;
        alpha(alpha<=-0.6) = -0.6;
        x=min(max(round(x+alpha(1)), 2), w-1);
        y=min(max(round(y+alpha(2)), 2), h-1);
    end
end
val = patch(2,2)-0.5*transpose(grad)*(hess\grad);
end

function [alpha, h, g] = quadraticInterpolate3D(patch)
g1=(patch(2,2,3)-patch(2,2,1))/2; % scale
g2=(patch(2,3,2)-patch(2,1,2))/2; % x
g3=(patch(3,2,2)-patch(1,2,2))/2; % y
g=[g1;g2;g3];

h11=patch(2,2,3)+patch(2,2,1)-2*patch(2,2,2);
h22=patch(2,3,2)+patch(2,1,2)-2*patch(2,2,2);
h33=patch(3,2,2)+patch(1,2,2)-2*patch(2,2,2);
h12=(patch(2,3,3)+patch(2,1,1)-patch(2,1,3)-patch(2,3,1))/4;
h13=(patch(3,2,3)+patch(1,2,1)-patch(1,2,3)-patch(3,2,1))/4;
h23=(patch(3,3,2)+patch(1,1,2)-patch(1,3,2)-patch(3,1,2))/4;
h=[h11,h12,h13;...
   h12,h22,h23;...
   h13,h23,h33];

alpha = -(h\g);
end

function [alpha, h, g] = quadraticInterpolate2D(patch)
g1=(patch(2,3)-patch(2,1))/2; % x
g2=(patch(3,2)-patch(1,2))/2; % y
g=[g1;g2];

h11 = patch(2,3)+patch(2,1)-2*patch(2,2);
h22 = patch(3,2)+patch(1,2)-2*patch(2,2);
h12 = (patch(1,1)+patch(3,3)-patch(1,3)-patch(3,1))/4;
h = [h11, h12; h12, h22];

alpha = -(h\g);
end

function [keypoints, DGS]=computeOrientation(keypoints, DGS)
global n_oct;
global n_spo;
lambda_ori=1.5;
nbin=36;
t=0.8;
[h,w] = size(DGS.octave(2).scale(1).image);
% compute image gradient
for o=1:n_oct
    for s=1:n_spo+2
        [gmag,gdir] = imgradient(DGS.octave(o).scale(s).image); % dir: -180~180
        DGS.octave(o).scale(s).gmag=gmag;
        DGS.octave(o).scale(s).gdir=gdir+180; % dir: 0~360
    end
end

delete_kp=[];
l=numel(keypoints);
for kp=1:l
    % get info of the current keypoint
    x=keypoints(kp).x_abs;
    y=keypoints(kp).y_abs;
    o=keypoints(kp).octave;
    s=round(keypoints(kp).s_dgs);
    d=DGS.octave(o).delta;
    sig=keypoints(kp).sigma;
    % check if the keypoint is distant enough from the image border
    if x <= 3*lambda_ori*sig+1 | x >= w-3*lambda_ori*sig | y <= 3*lambda_ori*sig+1 | y >= h-3*lambda_ori*sig
        delete_kp(end+1) = kp;
        continue;
    end
    
    mr=floor((x+3*lambda_ori*sig-1)/d+1);
    ml=ceil((x-3*lambda_ori*sig-1)/d+1);
    nb=floor((y+3*lambda_ori*sig-1)/d+1);
    nt=ceil((y-3*lambda_ori*sig-1)/d+1);
    % compute histogram
    hist=zeros(1,nbin);
    for xm=ml:mr
        for yn=nt:nb
            x_abs=1+d*(xm-1);
            y_abs=1+d*(yn-1);
            mag=DGS.octave(o).scale(s).gmag(yn, xm);
            dir=DGS.octave(o).scale(s).gdir(yn, xm);
            c=mag*exp(-sumsqr([x_abs-x, y_abs-y]/2/(lambda_ori*sig)^2));
            bin=round(nbin*dir/360)+1;
            if bin > nbin
                bin=1;
            end
            hist(bin)=hist(bin)+c;
        end
    end
    % smooth histogram
    for i=1:6
        hist = cconv(hist,[1/3, 1/3, 1/3],nbin);
    end
    % extract reference orientation
    num_dir = 0;
    max_bin = max(hist);
    % check head
    if (hist(1)>t*max_bin) & (hist(1)>hist(nbin)) & (hist(1)>hist(2))
        num_dir = num_dir+1;
        dir = 180/nbin*(hist(nbin)-hist(2))/(hist(nbin)-2*hist(1)+hist(2));
        if dir<0
            dir=dir+360;
        end
        keypoints(kp).ori=dir;
    end
    % check body
    for i=2:nbin-1
        if (hist(i)>t*max_bin) & (hist(i)>hist(i-1)) & (hist(i)>hist(i+1))
            num_dir = num_dir+1;
            dir = 360*(i-1)/nbin + 180/nbin*(hist(i-1)-hist(i+1))/(hist(i-1)-2*hist(i)+hist(i+1));
            if num_dir >=2
                keypoints(end+1) = keypoints(kp);
                keypoints(end).ori=dir;
            else
                keypoints(kp).ori=dir;
            end
        end
    end
    % check tail
    if (hist(nbin)>t*max_bin) & (hist(nbin)>hist(nbin-1)) & (hist(nbin)>hist(1))
        num_dir = num_dir+1;
        dir = 360*(nbin-1)/nbin + 180/nbin*(hist(nbin-1)-hist(1))/(hist(nbin-1)-2*hist(nbin)+hist(1));
        if dir>=360
            dir=dir-360
        end
        if num_dir >=2
            keypoints(end+1) = keypoints(kp);
            keypoints(end).ori=dir;
        else
            keypoints(kp).ori=dir;
        end
    end
end
% delete border keypoints
keypoints(delete_kp) = [];

end

function keypoints = removeScaleInvariance(keypoints)
l=numel(keypoints);
delete_kp = zeros(1,l);
parfor i=1:l
    if keypoints(i).octave ~= 2
        delete_kp(i) = 1;
    elseif keypoints(i).s_dgs >= 2.5 | keypoints(i).s_dgs < 1.5
        delete_kp(i) = 1;
    else
        keypoints(i).s_dgs = 2;
    end
end
keypoints(delete_kp==1)=[];
end

function keypoints = removeRotationInvariance(keypoints)
l=numel(keypoints);
parfor i=1:l
    keypoints(i).ori = 0;
end
end