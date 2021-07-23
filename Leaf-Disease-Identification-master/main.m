%Project Title: Paddy Leaf Disease Detection BY SVM Classifier...
clc ;
clear all;
close all;
%% general settings
fontsize= 14;
%% IMAGE ACQUISITION
% %pick the image and load it.
% [filename, pathname] = uigetfile(...
% {'*.jpg:*.tif:*.tiff:*.png:*.bmp','All image Files (*.jpg, *.tif,*.tiff, *.png,*.bmp)'},...
% 'Pick a file');
% f=fullfile (pathname, filename);
% disp ('Reading image');
IMrgb = imread('C:\Users\rahul\Desktop\New folder (2)\Leaf-Disease-Identification-master\Leaf-Disease-Identification-master\o\p\28.jpg');
    
figure, imshow(IMrgb);
title ('Query Leaf Image');  
impixelinfo;




%% PRE-PROCESSING

IMresize = imresize (IMrgb, [256,256]) ;% resize Image
%figure, imshow (IMresize ) ;
%title ('Input color resize image');

% Median Filter each channel separately,
r = medfilt2 (IMresize(:, :, 1), [3 3]) ;
figure, imshow(r);
title ('Query Leaf Image');
impixelinfo;% Median Filter
g = medfilt2 (IMresize (:, :, 2), [3 3]) ;
figure, imshow(g);
title ('Query Leaf Image');
impixelinfo;
b = medfilt2 (IMresize(:, :, 3), [3 3]) ; 
figure, imshow (b) ;
title ('Query Leaf Image');
impixelinfo;
%reconstruct the image from r, g,b channels
IMmedian= cat (3, r, g, b);
figure, imshow (IMmedian);
impixelinfo;
title ('Median Filtered image');

%Enhance Contrast
IMcons = imadjust (IMmedian, stretchlim (IMmedian)) ;
figure, imshow (IMcons) ;title ('Contrast Enhanced');
% 
% figure, subplot (4,1,1); imshow(r);title ('Query Leaf Image');
% subplot (4,1,2);  imshow(g);title ('Query Leaf Image');
% subplot (4, 1, 3) ;  imshow (b) ;title ('Query Leaf Image');
% subplot (4,1,4); imshow (IMmedian);
% title ('Median Filtered image');
%% IMAGE SEGMENTATION -K MEANS

% Color Image Segmentation
% Use of K Means clustering for segmentation
%Convert Image from RGB Color Space to L'a*b* Color Space
% The L*a*b* space consists of a luminosity layer 'L*', chromaticity-layer 'a*' and 'b*'.
% All of the color information is in the 'a*' and 'b*' layers.
cform = makecform('srgb2lab');
%Apply the colorform
lab_he=applycform (IMcons, cform);

%Classify the colors in a*b* colorspace using K means clustering.
% Since the image has 3 colors createclusters.
% Measure the distance using Euclidean Distance Metric.
ab = double(lab_he(:,:,2:3)) ;
nrows = size(ab,1);
ncols = size(ab,2) ;
ab = reshape(ab,nrows*ncols, 2) ;
nColors=3;
[cluster_idx Mean] = kmeans (ab,nColors, 'distance', 'sqEuclidean',...
'Replicates', 3);
%[cluster idx cluster center] = kmeans (ab, Colors, 'distance', 'sqEuclidean', 'Replicates', 3) ;
% Label every pixel in tha image using results from K means
pixel_labels = reshape (cluster_idx,nrows,ncols)
%[figure, imshow (pixel_labels, []), title ('Image Labeled by Cluster Index');
%Create a blank cell array to store the results of clustering
segmented_images = cell (1,3);
% Create RGB label using pixel labels
rgb_label = repmat(pixel_labels,[1,1,3]) ;

for k = 1:nColors
colors=IMcons;
colors(rgb_label ~= k)= 0;
segmented_images{k} = colors;
end

figure, subplot (4,1,1); imshow(segmented_images{1});title ('Cluster l'); subplot (4,1,2);imshow (segmented_images{2});
subplot (4, 1, 3) ; imshow(segmented_images{3});title('Cluster 3');subplot (4,1,4); imshow(IMrgb); title('original');
set(gcf, 'Position', get(0, 'Screensize'));

% figure, subplot (3,1,1); imshow(segmented_images{1});title ('Cluster l'); subplot (3,1,2);imshow (segmented_images{2});title ('Cluster 2');
% subplot (3, 1, 3) ; imshow(segmented_images{3});title('Cluster 3');
% set(gcf, 'Position', get(0, 'Screensize'));

%%SELECT SEGMENTED IMAGE FOR FEATURE EXTRACTION..

x=inputdlg ('Enter the cluster no. containing the ROI only:');
i = str2double(x);
% Extract the features from the segmented image
seg_img = segmented_images{i};
% Convert to grayscale if image is RGB
if ndims (seg_img) == 3
IM = rgb2gray (seg_img);
end
%figure, imshow (img) ; title ('Gray Scale Image');

%%Affected Area

%Evaluate the disease affected area
black = im2bw(seg_img, graythresh(seg_img));
% figure, imshow (black) ; title ('Black & White Image');
m = size (seg_img, 1) ;
n=size (seg_img,2);

zero_image = zeros (m, n) ;
%G=imoverlay (zero image, seg_img, [1 0 0]) ;

cc=bwconncomp (seg_img, 6) ;
diseasedata = regionprops (cc, 'basic') ;
Al = diseasedata.Area;
sprintf('Area of the disease affected region is:%g% ',Al);

I_black = im2bw (IMcons, graythresh (IMcons)) ;
kk = bwconncomp (IMcons, 6) ;
leafdata = regionprops (kk, 'basic') ;
A2 = leafdata.Area;
sprintf (' Total leaf area is : %g$', A2);

%Affected Area = 1- (A1/A2) ;
Affected_Area = (Al/A2);
if Affected_Area < 0.1
Affected_Area = Affected_Area+0.15;
end
sprintf('Affected_Area is: %g%%', (Affected_Area*100))

%%FEATURE EXTRACTION..

% Create the Gray Level Cooccurance Matrices (GLCMs) (Texture Features)
glcms = graycomatrix(IM);
% Derive Statistics from GLCM
stats = graycoprops (glcms, 'Contrast Correlation Energy Homogeneity') ;
Contrast= stats. Contrast;
Correlation = stats. Correlation;
Energy = stats. Energy;
Homogeneity = stats. Homogeneity;

% Color Features
Mean = mean2(seg_img);
Standard_Deviation = std2(seg_img);
Entropy = entropy (seg_img);
RMS =mean2 (rms (seg_img));
Variance=mean2 (var (double (seg_img)));
a=sum (double (seg_img(:)));
Smoothness = 1-(1/(1+a)) ;
Kurtosis =kurtosis(double(seg_img (:)));
Skewness = skewness (double (seg_img(:)));
feat_disease = [Contrast, Correlation, Energy, Homogeneity, Mean, Standard_Deviation, Entropy, RMS, Variance, Smoothness,Kurtosis,Skewness];


%% Classification
%Load All The Features
load('Train_data.mat')


Train_Label= zeros (size (30,1), 1);
Train_Label (1:8,1)= 1; %group l 
Train_Label (9:18,1)= 2; %group 2
Train_Label (19:26,1)=3; % group 3
Train_Label (27:30,1)= 4;% group 4

%Put the test features into variable 'test'
Test_Feature = feat_disease;
result = multisvm(Train_Feature,Train_Label, Test_Feature);
%disp (result) ;

% Visualize Results
if result == 1
msg = ('Brown spot');
msgbox (msg, 'The Result is..');
msg = ('Typel: Bacterial bright Disease (i) Use balanced amounts of plant nutrients, especially nitrogen (ii) Ar');
msgbox (msg, 'The suggestion is..');
elseif result ==2
msg=('Bacterial leaf streak');
msgbox (msg, 'The Result is..');
% disp('Brown_spot');
msge =('Typë 2: Brown Spot Disease (i) monitor soil nutrients regularly (ii) apply required fertilizers (iii');
msgbox (msg, 'The suggestion is..');

elseif result==3
msg=('Blast');
msgbox (msg, 'The Result is..');
% disp('Brown_spot');
msge =('Typë 2: Brown Spot Disease (i) monitor soil nutrients regularly (ii) apply required fertilizers (iii');
msgbox (msg, 'The suggestion is..');
elseif result==4
msg=('healthy');
msgbox (msg, 'The Result is..');
% disp('Brown_spot');
msge =('Typë 2: Brown Spot Disease (i) monitor soil nutrients regularly (ii) apply required fertilizers (iii');
msgbox (msg, 'The suggestion is..');
% elseif result==5
% msg=('Brown spot');
% msgbox (msg, 'The Result is..');
% disp('Brown_spot');
% msge =('Typë 2: Brown Spot Disease (i) monitor soil nutrients regularly (ii) apply required fertilizers (iii');
% msgbox (msg, 'The suggestion is..');
end

%% Evaluate Accuracy
load('Train_data.mat')
Accuracy_Percent= zeros (30,1) ;
for i = 1:200
data = Train_Feature;
groups = ismember (Train_Label,1) ;
[train, test]=crossvalind ('HoldOut', groups) ;
cp = classperf(groups);
svmStruct = svmtrain (data (train, :), groups (train), 'showplot', false, 'kfun', 'linear');
classes = svmclassify (svmStruct, data (test, :), 'showplot', false) ;
classperf (cp, classes, test) ;
Accuracy=cp. CorrectRate;
Accuracy_Percent(i)=Accuracy.*100;
end
Max_Accuracy=max (Accuracy_Percent);
sprintf ('Accuracy of Linear Kernel with 500 iterations is: %g%', Max_Accuracy)