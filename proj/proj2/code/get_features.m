% Local Feature Stencil Code
% CS 4476 / 6476: Computer Vision, Georgia Tech
% Written by James Hays

% Returns a set of feature descriptors for a given set of interest points. 

% 'image' can be grayscale or color, your choice.
% 'x' and 'y' are nx1 vectors of x and y coordinates of interest points.
%   The local features should be centered at x and y.
% 'feature_width', in pixels, is the local feature width. You can assume
%   that feature_width will be a multiple of 4 (i.e. every cell of your
%   local SIFT-like feature will have an integer width and height).
% If you want to detect and describe features at multiple scales or
% particular orientations you can add input arguments.

% 'features' is the array of computed features. It should have the
%   following size: [length(x) x feature dimensionality] (e.g. 128 for
%   standard SIFT)

function keypoints = get_features(keypoints, DGS)

% To start with, you might want to simply use normalized patches as your
% local feature. This is very simple to code and works OK. However, to get
% full credit you will need to implement the more effective SIFT descriptor
% (See Szeliski 4.1.2 or the original publications at
% http://www.cs.ubc.ca/~lowe/keypoints/)

% Your implementation does not need to exactly match the SIFT reference.
% Here are the key properties your (baseline) descriptor should have:
%  (1) a 4x4 grid of cells, each feature_width/4. 'cell' in this context
%    nothing to do with the Matlab data structue of cell(). It is simply
%    the terminology used in the feature literature to describe the spatial
%    bins where gradient distributions will be described.
%  (2) each cell should have a histogram of the local distribution of
%    gradients in 8 orientations. Appending these histograms together will
%    give you 4x4 x 8 = 128 dimensions.
%  (3) Each feature vector should be normalized to unit length
%
% You do not need to perform the interpolation in which each gradient
% measurement contributes to multiple orientation bins in multiple cells
% As described in Szeliski, a single gradient measurement creates a
% weighted contribution to the 4 nearest cells and the 2 nearest
% orientation bins within each cell, for 8 total contributions. This type
% of interpolation probably will help, though.

% You do not have to explicitly compute the gradient orientation at each
% pixel (although you are free to do so). You can instead filter with
% oriented filters (e.g. a filter that responds to edges with a specific
% orientation). All of your SIFT-like feature can be constructed entirely
% from filtering fairly quickly in this way.

% You do not need to do the normalize -> threshold -> normalize again
% operation as detailed in Szeliski and the SIFT paper. It can help, though.

% Another simple trick which can help is to raise each element of the final
% feature vector to some power that is less than one.

lambda_descr=6;
nhist=4;
nori=8;
[h,w] = size(DGS.octave(2).scale(1).image);

l=numel(keypoints);
keypoints(l).feature=zeros(nhist*nhist*nori);
delete_kp = zeros(1,l);
parfor kp=1:l
    x=keypoints(kp).x_abs;
    y=keypoints(kp).y_abs;
    o=keypoints(kp).octave;
    s=round(keypoints(kp).s_dgs);
    d=DGS.octave(o).delta;
    sig=keypoints(kp).sigma;
    ori=keypoints(kp).ori/180*pi;% convert to radian: 0~2pi
    % check border points
    if (x<=sqrt(2)*lambda_descr*sig+1 | x>=w-sqrt(2)*lambda_descr*sig |...
        y<=sqrt(2)*lambda_descr*sig+1 | y>=h-sqrt(2)*lambda_descr*sig)
        delete_kp(kp)=1;
        continue
    end
    
    hist=zeros(nhist,nhist,nori);
    ml=ceil((x-sqrt(2)*lambda_descr*sig-1)/d+1);
    mr=floor((x+sqrt(2)*lambda_descr*sig-1)/d+1);
    nt=ceil((y-sqrt(2)*lambda_descr*sig-1)/d+1);
    nb=floor((y+sqrt(2)*lambda_descr*sig-1)/d+1);
    for xm=ml:mr
        for yn=nt:nb
            % convert to normalied patch coordinate
            xnorm=(((xm-1)*d+1-x)*cos(ori)+((yn-1)*d+1-y)*sin(ori))/sig;
            ynorm=-(((xm-1)*d+1-x)*sin(ori)+((yn-1)*d+1-y)*cos(ori))/sig;
            % check within normalied patch
            if max(abs(xnorm), abs(ynorm)) >= lambda_descr*(nhist+1)/nhist
                continue;
            end
            % compute theta and contribution
            dir=DGS.octave(o).scale(s).gdir(yn,xm);
            dir_norm=mod(dir/180*pi-ori,2*pi);
            mag=DGS.octave(o).scale(s).gmag(yn,xm);
            xabs=(xm-1)*d+1;
            yabs=(yn-1)*d+1;
            c=mag*exp(-sumsqr([xabs-x, yabs-y]/2/(lambda_descr*sig)^2));
            % update histogram
            unit=2*lambda_descr/nhist;
            is=max(floor(xnorm/unit+2.5), 1);
            ie=min(ceil(xnorm/unit+2.5), 4);
            js=max(floor(ynorm/unit+2.5), 1);
            je=min(ceil(ynorm/unit+2.5), 4);
            for i=is:ie
                for j=js:je
                    xi=(i-(1+nhist)/2)*unit;
                    yj=(j-(1+nhist)/2)*unit;
                    dx=abs(xi-xnorm);
                    dy=abs(yj-ynorm);
                    
                    k=mod(round(dir_norm/(2*pi)*nori), nori)+1;
                    thetak1=2*pi*(k-1)/nori;
                    dtheta1=dir_norm-thetak1;
                    hist(i,j,k)=hist(i,j,k)+c*(1-dx/unit)*(1-dy/unit)*(1-dtheta1*nori/2/pi);
                    thetak2=2*pi*k/nori;
                    dtheta2=thetak2-dir_norm;
                    if k+1 > nori
                        k=1;
                    else
                        k=k+1;
                    end
                    hist(i,j,k)=hist(i,j,k)+c*(1-dx/unit)*(1-dy/unit)*(1-dtheta2*nori/2/pi);
                end
            end
        end
    end
    % build feature vector
    feature=zeros(1,nhist*nhist*nori);
    for i=1:nhist
        for j=1:nhist
            for k=1:nori
                ind=(i-1)*nhist*nori+(j-1)*nori+k;
                feature(ind)=hist(i,j,k);
            end
        end
    end
    % normalize feature vector
    feature=feature/norm(feature);
    feature(feature>0.2)=0.2;
    feature=feature/norm(feature);
    keypoints(kp).feature=feature;
end
keypoints(delete_kp==1)=[];

end