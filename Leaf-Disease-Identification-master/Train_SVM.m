clear all;

close all;
Train_Feature = [];


for imc=1:30
cd 'C:\Users\rahul\Desktop\New folder (2)\Leaf-Disease-Identification-master\Leaf-Disease-Identification-master\o\p'
    
f= [num2str(imc) '.jpg'];
IMrgb = imread(f);  %read image
cd ..

%%PRE-PROCESSING
IMresize =imresize (IMrgb, [256,256]) ;%  resize Image
%figure, imshow(IMresize);
%title ('Input color resize image');

% Median Filter each channel separately
r = medfilt2(IMresize (:, :, 1), [3 3]) ;%Median Filter
g=medfilt2 (IMresize (:, :, 2), [3 3]) ;
b=medfilt2 (IMresize (:, :, 3), [3 3]);
% reconstruct the image from r, g,b channels
IMmedian= cat (3, r, g, b);
% figure, imshow (IMmedian);
%impixelinfo;
%title ('Median Filtered image');

% Enhance Contrast
Icons = imadjust (IMmedian, stretchlim (IMmedian)) ;
%figure, imshow (Icons); title ('Contrast Enhanced');

%% IMAGE SEGMENTATION -K MEANS

%Color Image Segmentation
%Use of K Means clustering for segmentation
%Convert Image from RGB Color Space to L'a*b* Color Space
%The Lia*b* space consists of a luminosity layer 'L*', chromaticity-layer 'a*' and 'b*'.
%All of the color information is in the la*i and 'b*' layers.
cform = makecform ('srgb2lab') ;
%Apply the colorform
lab_he = applycform (Icons, cform) ;

%Classify the colors 1n a br colorspace using K means clustering.
% Since the image has 3 colors create 3 clusters.
% Measure the distance using Euclidean Distance Metric.
ab = double(lab_he (:,:,2:3)) ;
nrows = size (ab, 1) ;
ncols = size (ab, 2) ;
ab = reshape (ab, nrows*ncols, 2) ;
nColors=3;
[cluster_idx Mean]=kmeans (ab, nColors, 'distance', 'sqEuclidean',...
'Replicates', 3);
%[cluster_idx cluster center] = means (ab, nColors, 'distance', 'sqEuclidean', ' Replicates', 3);
%Label every pixel in tha image using results from K means
pixel_labels = reshape (cluster_idx,nrows,ncols);
%figure, imshow (pixel labels, []), title ('Image Labeled by Cluster Index');
%Create a blank cell array to store the results of clustering
segmented_images = cell (1,3);
%Create RGB label using pixel labels
rgb_label = repmat (pixel_labels, [1,1,3]) ;

for k = 1:nColors
colors =Icons;
colors (rgb_label ~= k) = 0;
segmented_images{k} = colors;
end

figure, subplot (4,1,1); imshow(segmented_images{1});title ('Cluster l'); subplot (4,1,2);imshow (segmented_images{2});
subplot (4, 1, 3) ; imshow(segmented_images{3});title('Cluster 3');subplot (4,1,4); imshow(IMrgb); title('original');
set(gcf, 'Position', get(0, 'Screensize'));

% figure, subplot (3,1,1); imshow(segmented_images{1}); title('Cluster 1'); subplot (3,1,2); imshow (segmented_images{2}) ;
% subplot (3, 1, 3) ; imshow (segmented_images {3}); title ('Cluster 3');
% set(gcf, 'Position', get(0, 'Screensize'));

%% SELECT SEGMENTED IMAGE FOR FEATURE EXTRACTION

x =inputdlg ('Enter the cluster no. containing the ROI only:');
i=str2double(x);
% Extract the features from the segmented image
seg_img = segmented_images{i};
%Convert to grayscale if image is RGB
if ndims (seg_img) == 3
IM = rgb2gray(seg_img);
end
%figure, imshow (img); title ('Gray Scale Image'):

%% FEATURE EXTRACTION
%Create the Gray Level Cooccurance Matrices (GLCMs) (Texture Features)
glcms = graycomatrix(IM);
% Derive Statistics from GLCM
stats = graycoprops (glcms, 'Contrast Correlation Energy Homogeneity') ;
Contrast = stats. Contrast;
Correlation = stats. Correlation;
Energy =stats. Energy;
Homogeneity = stats. Homogeneity;

%Color Features
Mean =mean2 (seg_img);
Standard_Deviation =std2 (seg_img)
Entropy = entropy(seg_img);
RMS = mean2(rms (seg_img));
Variance =mean2 (var (double (seg_img)));
a=sum (double (seg_img(:)));
Smoothness = 1-(1/(1+a)) ;
Kurtosis = - kurtosis (double (seg_img(:)));
Skewness = skewness (double (seg_img(:)));

Feature_Vector = [Contrast, Correlation, Energy, Homogeneity, Mean, Standard_Deviation, Entropy, RMS, Variance, Smoothness,Kurtosis,Skewness];
Train_Feature = [Train_Feature; Feature_Vector];

Train_Label= zeros (size (30,1),1);
Train_Label (1:8,1)= 1;% group 1
Train_Label (9:18,1)= 2;%group 2
Train_Label (19:26,1)= 3;%group 3
Train_Label (27:30,1)=4;%group 4
% Train_Label (62:65,1)=5;% group 5
end

save('Train_data.mat','Train_Label','Train_Feature')