train_pixels = 7500;
number_images = 6;
num_image = zeros([train_pixels*2*number_images,2]);
n = 1;
for j = 2:number_images+1
    path = strcat(string(j),'.png')
    labels = strcat(string(j),'.mat')
    Image = imread(path); %Converts the RGB into a grayscale
    mask = load(labels) %load the mask
    %find the shape
    %Divide row(r(i),1) to find the resizing factorn
    %Image = imresize(beginning,0.5,"bicubic");
    % sz = size(Image);
    Image = padarray(Image,[82 82],0,'both');
    % L=sz(1);
    % H=sz(2);
    % X = randi(L,15000,1)+82; %Coordinates but accounting for padding
    % Y=randi(H,15000,1)+82;
    test_image = zeros([33,33,3],'uint8');
    [row, column] = find(mask.BW==1);
    row = row+82;
    column = column +82; %to account for padding
    r = randperm(length(row(:,1)));
    train_pi = min([train_pixels length(r)])
    %% Train Vessels
    for i = 1:train_pi
        channel_1 = Image((row(r(i),1)-3):(row(r(i),1)+3),(column(r(i),1)-3):(column(r(i),1)+3));
        test_image(:,:,1) = imresize(channel_1,4.6,"bicubic");
        test_image(:,:,2) =  Image(row(r(i),1)-16:row(r(i),1)+16,column(r(i),1)-16:column(r(i),1)+16);
        channel_3 = Image(row(r(i),1)-82:row(r(i),1)+82,column(r(i),1)-82:column(r(i),1)+82);
        test_image(:,:,3) = imresize(channel_3,0.2,"bicubic");
        num_image(n,1)=1;
        imwrite(test_image,strcat("Image",string(n),".png"));
        n = n+1;
    end

    [row2, column2] = find(mask.BW==0);
    row2 = row2+82;
    column2 = column2 +82; %to account for padding
    r2 = randperm(length(row2(:,1)));
    train_pi = min([train_pixels length(r2)])
    for i=1:train_pi
        channel_1 = Image((row2(r2(i),1)-3):(row2(r2(i),1)+3),(column2(r2(i),1)-3):(column2(r2(i),1)+3));
        test_image(:,:,1) = imresize(channel_1,4.6,"bicubic");
        test_image(:,:,2) =  Image(row2(r2(i),1)-16:row2(r2(i),1)+16,column2(r2(i),1)-16:column2(r2(i),1)+16);
        channel_3 = Image(row2(r2(i),1)-82:row2(r2(i),1)+82,column2(r2(i),1)-82:column2(r2(i),1)+82);
        test_image(:,:,3) = imresize(channel_3,0.2,"bicubic");
        num_image(n,2)=1; % one hot encoder
        imwrite(test_image,strcat("Image",string(n),".png"));
        n = n+1;
    end

end

save("labels.mat","num_image")
toc
num_image2 =zeros([test_pixels*2,2]);
test_pixels = 1000;
%% Test Vessels
tic
for i = train_pixels+1:train_pixels+test_pixels
    channel_1 = Image((row(r(i),1)-3):(row(r(i),1)+3),(column(r(i),1)-3):(column(r(i),1)+3));
    test_image(:,:,1) = imresize(channel_1,4.6,"bicubic");
    test_image(:,:,2) =  Image(row(r(i),1)-16:row(r(i),1)+16,column(r(i),1)-16:column(r(i),1)+16);
    channel_3 = Image(row(r(i),1)-82:row(r(i),1)+82,column(r(i),1)-82:column(r(i),1)+82);
    test_image(:,:,3) = imresize(channel_3,0.2,"bicubic");
    num_image2(i-train_pixels,1)=1;
    imwrite(test_image,strcat("Image",string(i-train_pixels),".png")); 
end



for i = train_pixels+1:train_pixels+test_pixels
    channel_1 = Image((row2(r2(i),1)-3):(row2(r2(i),1)+3),(column2(r2(i),1)-3):(column2(r2(i),1)+3));
    test_image(:,:,1) = imresize(channel_1,4.6,"bicubic");
    test_image(:,:,2) =  Image(row2(r2(i),1)-16:row2(r2(i),1)+16,column2(r2(i),1)-16:column2(r2(i),1)+16);
    channel_3 = Image(row2(r2(i),1)-82:row2(r2(i),1)+82,column2(r2(i),1)-82:column2(r2(i),1)+82);
    test_image(:,:,3) = imresize(channel_3,0.2,"bicubic");
    num_image2(i-train_pixels+test_pixels,2)=1; % one hot encoder
    imwrite(test_image,strcat("Image",string(i-train_pixels+test_pixels),".png"));
end
save("labels.mat","num_image2")
toc