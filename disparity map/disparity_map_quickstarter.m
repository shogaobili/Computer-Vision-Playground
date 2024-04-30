clear all
close all;

%% Read the images

imgLeft = imread("left.png");
imgRight = imread("right.png");

%% Setting up the hyperparams

disparityMax = 50; % Max disparity value
windowSize = 21; % Search window size
slidingSize = 50;

%% Convert the image in grayscale
leftSide = rgb2gray(imgLeft);
rightSide = rgb2gray(imgRight);

%% Some sanity checks
if ~all(size(leftSide) == size(rightSide))
    error("Images have different sizes");
end


%% Calculate disparity map
% [height, width] = size(leftSide);
% disparityMap = single(zeros([height,width])); 
% h = floor(windowSize/2); % get half of the window size
% 
% hWaitBar = waitbar(0, 'Calculating disparity map...');
% totalIterations = (height - 2 * h) * (width - 2 * h);
% currentIteration = 0;
% 
% %%
% % Your code here. Good luck :)
% for i = h+1:height-h
%     for j = h+1:width-h
%         % Select block in the left image
%         disp(i-h);
%         disp(i+h);
%         disp(j-h);
%         disp(j+h);
%         disp("-----------");
%         blockLeft = leftSide(i-h:i+h, j-h:j+h);
% 
%         % Initialize minimum sum of absolute differences (SAD) and disparity
%         minSAD = inf;
%         disparity = disparityMax;
%         % Search for matching block in the right image
%         for k = max(j-slidingSize, h + 1):min(j+slidingSize, width-h)
%             blockRight = rightSide(i-h:i+h, k-h:k+h);
% 
%             % Calculate the sum of absolute differences (SAD)
%             SAD = sum(abs(blockLeft(:) - blockRight(:)));
% 
%             % Update minimum SAD and disparity if current SAD is smaller
%             if SAD < minSAD
%                 minSAD = SAD;
%                 disparity = abs(j-k);
%             end
%         end
% 
%         % Set the disparity value for the current pixel
%         disparityMap(i, j) = disparity;
%         % Update waitbar
%         currentIteration = currentIteration + 1;
%         waitbar(currentIteration/totalIterations, hWaitBar);
%     end
% end
% close(hWaitBar);

disparityMap = disparitySGM(leftSide, rightSide);

%% Show result

% if it's not clear for you why I normalise the image below, come and ask me.
disparityMap_display = disparityMap / max(disparityMap(:));

imshow(disparityMap_display);
colormap jet;