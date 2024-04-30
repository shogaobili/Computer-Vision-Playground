% HoG feature learing example code
% Based on a nice tutorial at https://uk.mathworks.com/help/vision/examples/digit-classification-using-hog-features.html

close all;
clear all;

%% Load training and test data using |imageDatastore|. These come with Matlab.
% You could try adapting to your own image sets?
syntheticDir   = fullfile('C:\Users\34890\Desktop\CV week2\myDigits');
handwrittenDir = fullfile(toolboxdir('vision'), 'visiondata','digits','handwritten');

% |imageDatastore| recursively scans the directory tree containing the
% images. Folder names are automatically used as labels for each image.
trainingSet = imageDatastore(syntheticDir,   'IncludeSubfolders', true, 'LabelSource', 'foldernames');
testSet     = imageDatastore(handwrittenDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

%% tabulate the number of images with each label
countEachLabel(trainingSet)
countEachLabel(testSet)

%% Show some example images from the training set. Could you also show some from the trianing set?
figure;
subplot(2,3,1);
imshow(trainingSet.Files{12});
subplot(2,3,2);
imshow(trainingSet.Files{34});
subplot(2,3,3); % TEST SET
imshow(testSet.Files{5});
title('Example images')

%% Show pre-processing results
% This is just for display. Actual processing happens further down in a
% loop.
% (Q2) How does other processing affect results? Don't forget you 
exTestImage = readimage(testSet,120); % Change the number 
processedImage = imbinarize(rgb2gray(exTestImage));


figure;
subplot(1,2,1)
imshow(exTestImage)
subplot(1,2,2)
imshow(processedImage)
title('Processed results')

img = readimage(trainingSet, 26); % Pick a random training image as an example

%% Extract HOG features and HOG visualization (Q3: see how cell size affects the result?)
[hog_2x2, vis2x2] = extractHOGFeatures(img,'CellSize',[3 3]);
[hog_4x4, vis4x4] = extractHOGFeatures(img,'CellSize',[4 4]);
[hog_8x8, vis8x8] = extractHOGFeatures(img,'CellSize',[6 6]);

%% Show the original image
figure; 
subplot(2,3,1:3); imshow(img);

%% Visualize the HOG features
subplot(2,3,4);  
plot(vis2x2); 
title({'CellSize = [2 2]'; ['Length = ' num2str(length(hog_2x2))]});
subplot(2,3,5);
plot(vis4x4); 
title({'CellSize = [4 4]'; ['Length = ' num2str(length(hog_4x4))]});
subplot(2,3,6);
plot(vis8x8); 
title({'CellSize = [8 8]'; ['Length = ' num2str(length(hog_8x8))]});

%% From the images, this is a good compromise. Q3: What happens if you change it?
cellSize = [4 4];
hogFeatureSize = length(hog_4x4);



%% Loop over the trainingSet and extract HOG features from each image. A
% similar procedure will be used to extract features from the testSet.
numImages = numel(trainingSet.Files);
trainingFeatures = zeros(numImages, hogFeatureSize, 'single');

for i = 1:numImages
    img = readimage(trainingSet, i);
    
    img = rgb2gray(img);
    
    % Apply pre-processing steps. (Q2) PUT YOUR PRE PROCESSING HERE 
    img = imbinarize(img);
    
    trainingFeatures(i, :) = extractHOGFeatures(img, 'CellSize', cellSize);  
end

% Get labels for each image.
trainingLabels = trainingSet.Labels;

%% fitcecoc uses SVM learners and a 'One-vs-One' encoding scheme.
% This one line actually trains the classifier
classifier = fitcecoc(trainingFeatures, trainingLabels);

%% Let's test how well it works

% Extract HOG features from the test set. The procedure is similar to what
% was shown earlier and is encapsulated as a helper function for brevity.

% See end of file for functions
[testFeatures, testLabels] = helperExtractHOGFeaturesFromImageSet(testSet, hogFeatureSize, cellSize);



%% Make class predictions using the test features.
predictedLabels = predict(classifier, testFeatures);

%% Tabulate the results using a confusion matrix.
confMat = confusionmat(testLabels, predictedLabels);

helperDisplayConfusionMatrix(confMat)


function helperDisplayConfusionMatrix(confMat)
% Display the confusion matrix in a formatted table.

% Convert confusion matrix into percentage form
confMat = bsxfun(@rdivide,confMat,sum(confMat,2));

digits = '0':'9';
colHeadings = arrayfun(@(x)sprintf('%d',x),0:9,'UniformOutput',false);
format = repmat('%-9s',1,11);
header = sprintf(format,'digit  |',colHeadings{:});
fprintf('\n%s\n%s\n',header,repmat('-',size(header)));
for idx = 1:numel(digits)
    fprintf('%-9s',   [digits(idx) '      |']);
    fprintf('%-9.2f', confMat(idx,:));
    fprintf('\n')
end
end


function [features, setLabels] = helperExtractHOGFeaturesFromImageSet(imds, hogFeatureSize, cellSize)
% Extract HOG features from an imageDatastore.

setLabels = imds.Labels;
numImages = numel(imds.Files);
features  = zeros(numImages, hogFeatureSize, 'single');

% Process each image and extract features
for j = 1:numImages
    img = readimage(imds, j);
    img = rgb2gray(img);
    
    % Apply pre-processing steps
    img = imbinarize(img);  %Q2 and also add preprocessing here (for the test dataset)
    
    features(j, :) = extractHOGFeatures(img,'CellSize',cellSize);
end
end





