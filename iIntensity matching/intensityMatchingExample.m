% clear; close all; clc
% 
% %Read in images
% fixed = imread('cactus3.png'); %does it need resizing?
% fixed = rgb2gray(fixed);
% 
% moving  = imread('cactus3_crop1.png');
% moving = rgb2gray(moving);
% 
% % Setting the size of the crop - could read from the image alternatively.
% sz=100; %template size (moving image). 
% 
% %% Normalise image and tmplt by dividing by 255
% % Do Euclidean distance based-matching (SSD) between the template and image
% img = double(fixed(:, :, 1))./255;
% moving = double(moving)./255;
% 
% %%
% score = zeros(size(img, 1)-sz, size(img, 2)-sz); % store the error map
% 
% for ii = 1:size(img, 1)-sz
%     for jj = 1:size(img, 2)-sz
%         tar = img(ii:ii+sz-1, jj:jj+sz-1);     
%         score(ii, jj) = sum((moving(:) - tar(:)).^2); %Squared difference
%     end 
% end
% 
% %Show output figure - error map
% figure; imagesc(score); colorbar
% 
% %find min position from calculated scores
% [posx, posy] = find(score == min(min(score)));
% best_match_ED = [posx posy];
% 
% %Show rectangle on error map
% hold on 
% rectangle('Position',[posy, posx, sz, sz],'LineWidth',2, 'EdgeColor', 'r')
% hold off
% 
% %Show rectangle on original image
% figure; imshow(fixed);
% hold on 
% rectangle('Position',[posy, posx, sz, sz],'LineWidth',2, 'EdgeColor', 'r')
% hold off

%%
% Now replace Euclidean Distance with cross-correlation based-matching between the template and image
% See https://en.wikipedia.org/wiki/Cross-correlation
% Plot the correlation surface as a 2D image and find the location that
% corresponds to the max value
% What do you observe? Does it work well or not?
% img = double(fixed(:, :, 1))./255;
% moving = double(moving)./255;
% score = zeros(size(img, 1)-sz, size(img, 2)-sz);
% for ii = 1:size(img, 1)-sz
%     for jj = 1:size(img, 2)-sz
%         tar = img(ii:ii+sz-1, jj:jj+sz-1);
%         score(ii, jj) = moving(:)'*tar(:);
%     end 
% end
% 
% figure; imagesc(score); colorbar
% 
% %find max position from calculted scores
% [posx, posy] = find(score == max(max(score)));
% 
% % Just verify the same calculations as above using 2D cross correlation 
% % function in Signal Processing Toolbox if installed
% xcormap = xcorr2(img,moving);
% max = max(max(abs(xcormap)));
% [posx2, posy2] = find(xcormap == max);
% 
% best_match_Corr = [posx posy];
% best_match_Corr2 = [posx2-sz posy2-sz]; % Signal processing function version
% 
% 
% figure,imshow(img,[]);
% hold on 
% rectangle('Position',[posy, posx, sz, sz],'LineWidth',2, 'EdgeColor', 'r')
% hold off
% 
% figure; imshow(fixed);
% hold on 
% rectangle('Position',[posy, posx, sz, sz],'LineWidth',2, 'EdgeColor', 'r')
% hold off



% Repeat the above but replace correlation with *zero-normalised* cross-correlation


img = double(fixed(:, :, 1))./255;
moving = double(moving)./255;
score = zeros(size(img, 1)-sz, size(img, 2)-sz);
tmplt1 = moving(:) - mean(moving(:)); 
tmplt1 = tmplt1./norm(tmplt1);

disp(tmplt1)
%%
for ii = 1:size(img, 1)-sz
    for jj = 1:size(img, 2)-sz
        tar = img(ii:ii+sz-1, jj:jj+sz-1);
        tar = tar(:) - mean(tar(:)); 
        tar = tar./norm(tar);
        score(ii, jj) = tmplt1(:)'*tar(:);
    end 
end
[posx, posy] = find(score == max(max(score)));
figure; imagesc(score); colorbar
%figure; imagesc(squeeze(data(:,:,1))); colormap(gray)
best_match_Corr = [posx posy];
hold on 
rectangle('Position',[posy, posx, sz, sz],'LineWidth',2, 'EdgeColor', 'r')
hold off

figure; imshow(fixed);
hold on 
rectangle('Position',[posy, posx, sz, sz],'LineWidth',2, 'EdgeColor', 'r')
hold off


%% Extra task
%% Try using HOG features
%% Could you adapt the above code to use HoG features?