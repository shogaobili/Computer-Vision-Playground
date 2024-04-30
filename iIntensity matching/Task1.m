clear; close all; clc

fixed = imread("cactus4.png");
moving = imread("cactus5.png");

fixed = rgb2gray(fixed);
moving = rgb2gray(moving);

fixedResized = imresize(fixed, [640 640]);
movingResized = imresize(moving, [640 640]);

subplot(1, 2, 1);
imshowpair(movingResized, fixedResized);
title('Before Registration');

[optimizer, metric]  = imregconfig('monomodal');
optimizer.MaximumIterations = 3000;

movingRegistered = imregister(moving, fixed, 'affine', optimizer, metric);
movingRegisteredResized = imresize(movingRegistered, [640 640]);

subplot(1, 2, 2);
imshowpair(movingRegisteredResized, fixedResized);
title('After Registration');