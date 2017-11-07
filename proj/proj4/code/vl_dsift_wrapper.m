function SIFT_features = vl_dsift_wrapper(I, step, split, pyr)
for i=1:pyr
    features = cell(split*split,1);
    for j=1:split %x
        for k=1:split %y
            if i==1
                Ir = I;
            else
                If = imgaussfilt(I, 2^(i-1));
                Ir = imresize(If, 1/2^(i-1));
            end
            [h,w]=size(Ir);
            xmin=round(1+(j-1)*w/split);
            xmax=round(j*w/split);
            ymin=round(1+(k-1)*h/split);
            ymax=round(k*h/split);
            Icrop = Ir(ymin:ymax, xmin:xmax);
            [~, feature] = vl_dsift(Icrop, 'Step', step, 'Fast');
            features{} = feature;
        end
    end
end


end

