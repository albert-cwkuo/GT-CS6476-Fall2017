function output = my_imfilter(image, filter)
% This function is intended to behave like the built in function imfilter()
% See 'help imfilter' or 'help conv2'. While terms like "filtering" and
% "convolution" might be used interchangeably, and they are indeed nearly
% the same thing, there is a difference:
% from 'help filter2'
%    2-D correlation is related to 2-D convolution by a 180 degree rotation
%    of the filter matrix.

% Your function should work for color images. Simply filter each color
% channel independently.

% Your function should work for filters of any width and height
% combination, as long as the width and height are odd (e.g. 1, 7, 9). This
% restriction makes it unambigious which pixel in the filter is the center
% pixel.

% Boundary handling can be tricky. The filter can't be centered on pixels
% at the image boundary without parts of the filter being out of bounds. If
% you look at 'help conv2' and 'help imfilter' you see that they have
% several options to deal with boundaries. You should simply recreate the
% default behavior of imfilter -- pad the input image with zeros, and
% return a filtered image which matches the input resolution. A better
% approach is to mirror the image content over the boundaries for padding.

% % Uncomment if you want to simply call imfilter so you can see the desired
% % behavior. When you write your actual solution, you can't use imfilter,
% % filter2, conv2, etc. Simply loop over all the pixels and do the actual
% % computation. It might be slow.
% gt = imfilter(image, filter);


%%%%%%%%%%%%%%%%
% Your code here
%%%%%%%%%%%%%%%%
filter_size = size(filter);
pad_size = floor(filter_size/2);
image_padded = padarray(image, pad_size, 'symmetric');

image_size = size(image);
output = zeros(image_size);
% uncomment to measure the running time
% disp('Arithmetic expands');
% tic
for h=1:image_size(1)
    for w=1:image_size(2)
        receptive_field = image_padded(h:h+2*pad_size(1), w:w+2*pad_size(2), 1:end);
        filtered = receptive_field .* filter;
        output(h,w,1:end) = sum(sum(filtered, 1), 2);
    end
end
% toc

% uncomment to measure the running time of traditional method
% disp('Non arithmetic expands');
% tic
% for h=1:image_size(1)
%     for w=1:image_size(2)
%         for c=1:image_size(3)
%             receptive_field = image_padded(h:h+2*pad_size(1), w:w+2*pad_size(2), c);
%             filtered = receptive_field .* filter;
%             output(h,w,c) = sum(sum(filtered, 1), 2);
%         end
%     end
% end
% toc