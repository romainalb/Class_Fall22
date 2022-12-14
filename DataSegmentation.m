videoReader = VideoReader('2022-02-01-stroke.avi');
allFrames = read(videoReader);
middle = allFrames(:,:,:,7860);
beginning = allFrames(:,:,:,60);
end_frame = allFrames(:,:,:,11580);
videoReader.VideoFormat
videoFrame = readFrame(videoReader);
videoFrame = rgb2gray(videoFrame); %Converts the RGB into a grayscale
figure 
imshow(videoFrame);
reduced_image = imresize(beginning,0.5,"bicubic");
figure 
imshow(reduced_image);
imageSegmenter(reduced_image);
imshow(beginning,[1 120])
figure
imshow(BW);
figure
imshow(maskedImage);
savebeginning
%test = onehotencode(BW) %This require the latest version of Matlab 
%%I need to augmente the data so that it can fit the edge (at least by 82
%%on each direction)
reduced_image = rgb2gray(reduced_image)
Y=409
X=261

tic
test_image = zeros([33,33,3],'uint8');
channel_1 = reduced_image((X-3):(X+3),(Y-3):(Y+3));
test_image(:,:,1) = imresize(channel_1,4.6,"bicubic");
test_image(:,:,2) =  reduced_image(X-16:X+16,Y-16:Y+16);
channel_3 = reduced_image(X-82:X+82,Y-82:Y+82);
test_image(:,:,3) = imresize(channel_3,0.2,"bicubic");
imwrite(test_image,'test_image.png');
toc


b = imread('test_image.png');
figure
imshow(b(:,:,1))
figure
imshow(b(:,:,2))
figure
imshow(b(:,:,3))

channel_2 = reduced_image(X-16:X+16,Y-16:Y+16);
channel_3 = reduced_image(X-82:X+82,Y-82:Y+82);
channel_3_small = imresize(channel_3,0.2,"bicubic");

figure
imshow(channel_1_big);
figure
imshow(channel_2);
figure
imshow(channel_3_small);

%% Evaluation over time 
middle = allFrames(:,:,:,7860);
middle = rgb2gray(middle);
beginning = allFrames(:,:,:,60);
beginning = rgb2gray(beginning);
end_frame = allFrames(:,:,:,11580);
end_frame = rgb2gray(end_frame);

imageSegmenter(beginning);
imshow(Middle, 'InitialMag', 'fit') 

imshow(middle, 'InitialMag', 'fit') 
% Make a truecolor all-green image. 
green = cat(3, ones(size(middle)),... 
    zeros(size(middle)), zeros(size(middle))); 
hold on 

hold off 
set(h, 'AlphaData', Middle) 

figure
imshow(middle);
% superimpose a contour map (in this case of the image brightness)
hold on;
contour(Middle, 5,'r');
contour(End_BackToBetter, 5,'b');
hold off;
%% Overlay the difference between beginning and end
imshow(middle, 'InitialMag', 'fit') 
% Make a truecolor all-green image. 
green = cat(3, ones(size(middle)),... 
    zeros(size(middle)), zeros(size(middle))); 
yellow = cat(3, ones(size(middle)),... 
    ones(size(middle)), zeros(size(middle))); 
hold on
h = imshow(green);
j = imshow(yellow); 
hold off 
set(j, 'AlphaData', End_BackToBetter) 
set(h, 'AlphaData', Middle) 

