clc;
clear all;
close all;

% wczytanie repozytoriów
masks = dir(fullfile('D:','Studia','VI','POM','posegmentowane binary','binaries'));
origs = dir(fullfile('D:','Studia','VI','POM','posegmentowane binary','oryg'));

masks([1,2],:)=[];
origs([1,2],:)=[];
% rozmiar docelowy
targetSize  = [400 400];

NF = length(masks);

mask_cell = cell(NF,1);
im_cell = cell(NF,1);

% augmetnacja danych
imageAugmenter = imageDataAugmenter('RandRotation',[-20, 20], 'RandXTranslation', [-44,44], 'RandYTranslation', [-44,44]);

for k = 1 : NF
  binary = imread(fullfile('binaries', masks(k).name));
  image = imread(fullfile('oryg', origs(k).name));

  one_misshaped = im2uint8(binary); %nastêpuje konwersja z BW do uint8
  one = imresize(one_misshaped, [400,400]); %resize
  
  mask_cell{k} = cat(3,one, one, one); % z jednowymiarowej macierzy do RGB
  im_cell{k} = imresize(image, [400,400]); %resize

  augi = augment(imageAugmenter, {im_cell{k},mask_cell{k}});

  imaug = augi{1};
  maskaug = augi{2};

    iter = 6; % ile iteracji augmentacji danych 
% zapis
  imwrite(imaug, strcat("D:\Studia\VI\POM\CNNsegmentation\resized\augs\imgs\o_", int2str(k+iter*77), ".png"));
  imwrite(maskaug, strcat("D:\Studia\VI\POM\CNNsegmentation\resized\augs\masks\m_", int2str(k+iter*77), ".png"));

end















%% TEGO NIE ROBIÆ ABSOLUTNIE, BO ZAMIAST PRZYCINANIA JEST RESIZE PARAMI
% 
% 
% mask_cropped = cell(NF,1);
% orig_cropped = cell(NF,1);
% 
% for a=1:NF
%     disp(a)
% m = centerCropWindow2d(size(mask_cell{a}), targetSize); 
% o = centerCropWindow2d(size(im_cell{a}), targetSize); 
% 
% mm = imcrop(mask_cell{a},m); 
% oo = imcrop(im_cell{a},o); 
% 
% mask_cropped{a}=mm;
% orig_cropped{a}=oo;    %tu by³o rgb2gray ale wyrzuci³em
% 
% imwrite(mask_cropped{a}, strcat("cropped_masks\m_c_", int2str(a), ".png"));
% imwrite(orig_cropped{a}, strcat("cropped_origs\o_c_", int2str(a), ".png"));
% end
%% reszta













