% Starter code prepared by James Hays for Computer Vision

%This function will train a linear SVM for every category (i.e. one vs all)
%and then use the learned linear classifiers to predict the category of
%every test image. Every test feature will be evaluated with all 15 SVMs
%and the most confident SVM will "win". Confidence, or distance from the
%margin, is W*X + B where '*' is the inner product or dot product and W and
%B are the learned hyperplane parameters.

function predicted_categories = svm_classify(train_image_feats, train_labels, test_image_feats, linear, retrain)
% train_image_feats is an N x d matrix, where d is the dimensionality of the
%  feature representation.
% train_labels is an N x 1 cell array, where each entry is a string
%  indicating the ground truth category for each training image.
% test_image_feats is an M x d matrix, where d is the dimensionality of the
%  feature representation. You can assume M = N unless you've modified the
%  starter code.
% predicted_categories is an M x 1 cell array, where each entry is a string
%  indicating the predicted category for each test image.

%{
Useful functions:
 matching_indices = strcmp(string, cell_array_of_strings)
 
  This can tell you which indices in train_labels match a particular
  category. This is useful for creating the binary labels for each SVM
  training task.

[W B] = vl_svmtrain(features, labels, LAMBDA)
  http://www.vlfeat.org/matlab/vl_svmtrain.html

  This function trains linear svms based on training examples, binary
  labels (-1 or 1), and LAMBDA which regularizes the linear classifier
  by encouraging W to be of small magnitude. LAMBDA is a very important
  parameter! You might need to experiment with a wide range of values for
  LAMBDA, e.g. 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10.

  Matlab has a built in SVM, see 'help svmtrain', which is more general,
  but it obfuscates the learned SVM parameters in the case of the linear
  model. This makes it hard to compute "confidences" which are needed for
  one-vs-all classification.

%}

%unique() is used to get the category list from the observed training
%category list. 'categories' will not be in the same order as in proj4.m,
%because unique() sorts them. This shouldn't really matter, though.
categories = unique(train_labels); 

C = numel(categories);
N = size(train_image_feats,1);
M = size(test_image_feats,1);
if linear % linear SVM
    W = cell(C,1);
    B = cell(C,1);
    LAMBDA = 0.000001;
    parfor i=1:C
        L = double(strcmp(categories{i}, train_labels));
        L(L==0) = -1;
        [w, b] = vl_svmtrain(train_image_feats', L, LAMBDA);
        W{i}=w';
        B{i}=b;
    end
    Wmat = cell2mat(W);
    Bmat = cell2mat(B);

    cls = Wmat*test_image_feats'+Bmat;
    [~,ind] = max(cls);

    predicted_categories = cell(M,1);
    parfor i=1:M
        predicted_categories{i} = categories{ind(i)};
    end
else% non-linear svm
    randL = rand(M,1);
    L=zeros(N,1);
    for i=1:C
        L(strcmp(categories{i}, train_labels)) = i;
    end
    options='-s 0 -t 2 -c 200 -b 1 -g 0.001 -q';
    if ~exist('model.mat', 'file') | retrain
        model=svmtrain(L, train_image_feats, options);
        save('model.mat', 'model')
    else
        load('model.mat', 'model');
        model=model;
    end
    [pred,~ ,~] = svmpredict(randL,test_image_feats, model,'-b 1');
    
    predicted_categories = cell(M,1);
    parfor i=1:M
        predicted_categories{i} = categories{pred(i)};
    end
end

end



