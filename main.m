% clc;
% close all;
% clear all;

%% Przygotowanie repozytoriów obrazów i ich masek
 
vgg16;

labelDir = fullfile('D:','Studia','VI','POM','CNNsegmentation','resized','augs','masks');
imgDir = fullfile('D:','Studia','VI','POM','CNNsegmentation','resized','augs','imgs');

% HelperFunctions.prepareData(masksDir, origsDir); 
% tutaj siê dzieje
% preprossessing i resize

imds = imageDatastore(imgDir);

pic_num=345; % jest 77 ale 5 bêdzie do testów
I_raw = readimage(imds, pic_num);

classes = ["Cancer", "Rest"];
labelIDs = HelperFunctions.camvidPixelLabelIDs();

pxds = pixelLabelDatastore(labelDir, classes, labelIDs);

cmap = HelperFunctions.camvidColorMap;

HelperFunctions.showImageMapping(pxds, cmap);

drawnow

C = readimage(pxds, pic_num); 
B = labeloverlay(I_raw, C, 'ColorMap', cmap);

figure
imshow(B)
drawnow
HelperFunctions.pixelLabelColorbar(cmap, classes);

[imdsTrain, imdsTest, pxdsTrain, pxdsTest] = HelperFunctions.partitionCamVidData(imds, pxds, labelIDs);

numTrainingImages = numel(imdsTrain.Files);
numTestingImages = numel(imdsTest.Files);
%

tbl = countEachLabel(pxds);

frequency = tbl.PixelCount/sum(tbl.PixelCount);
figure
bar(1:numel(classes),frequency);
xticks(1:numel(classes));
xticklabels(tbl.Name);
xtickangle(45);
ylabel('frequency')
%% Tworzenie i modyfikacja sieci neuronowej

imageSize = [400,400,3];
numClasses = numel(classes);

lgraph = segnetLayers(imageSize, numClasses,'vgg16');
lgraph.Layers;

% graf struktura
fig1 = figure('Position', [100, 100, 1000, 1100]);

subplot(1,2,1);
plot(lgraph);
axis off
axis tight 
title("Complete layer graph");

subplot(1,2,2);
plot(lgraph);
xlim([2.862, 3.200]);
ylim=([-0.9, 10.9]);
axis off
title("Last 9 layers graph")

imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classWeights = median(imageFreq) ./ imageFreq;

pxLayer = pixelClassificationLayer('Name', 'labels', 'ClassNames', tbl.Name, 'ClassWeights', classWeights);

% porównanie struktur
fig2 = figure('Position', [100, 100, 800, 600]);
subplot(1,2,1);
plot(lgraph);
xlim([2.862, 3.200]);
ylim=([-0.9, 10.9]);
axis off
axis tight 
title("Initial last 9 layers of the graph");

%usuñ ostatni¹ warstwê i dodaj now¹
lgraph = removeLayers(lgraph, {'pixelLabels'});
lgraph = addLayers(lgraph, pxLayer);
%po³¹cz z grafem
lgraph = connectLayers(lgraph, 'softmax', 'labels');
lgraph.Layers

% porównanie struktur
subplot(1,2,2)
plot(lgraph)
xlim([2.862, 3.200]);
ylim=([-0.9, 10.9]);
axis off
title("Modified last 9 layers of the graph");

% parametry uczenia
options = trainingOptions('sgdm',...
    'Momentum', 0.9, ...
    'InitialLearnRate', 1e-2, ...
    'L2Regularization', 0.0005, ...
    'MaxEpochs', 120, ...
    'MiniBatchSize', 1, ...
    'Shuffle', 'every-epoch',...
    'Verbose', false, ...
    'Plots', 'training-progress');

datasource = pixelLabelImageSource(imdsTrain,pxdsTrain);

doTraining = false; 
if doTraining
    tic
    [net, info]=trainNetwork(datasource, lgraph, options);
    toc
    save('PreTrainedCNN.mat','net','info','options');
    disp('NN trained');
else
    data = load('PreTrainedCNN.mat');
    net=data.net
end    
   
 %% Próba dzia³ania 
 
%  
% pic_num = 345;
pic_num = 34; %przyk³adowe zdjêcie
I = readimage(imds, pic_num);
Ib = readimage(pxds, pic_num);

mycmap = [144/255,238/255,144/255;0,0,0];

IB = labeloverlay(I,Ib,'Colormap',mycmap,'Transparency',0.8);
figure
C = semanticseg(I,net);
CB = labeloverlay(I,C,'Colormap',mycmap,'Transparency',0.8);
figure
imshowpair(IB,CB,'montage');
HelperFunctions.pixelLabelColorbar(mycmap,classes);
title('Ground truth vs Predicted');

%% Liczenie wsp Jaccarda z podzia³em na rozmiar - wczytanie folderów
clc;
smol = dir(fullfile('D:','Studia','VI','POM','CNNsegmentation','resized','sortedBySize','smol','img'));
smol([1,2],:)=[];
regular = dir(fullfile('D:','Studia','VI','POM','CNNsegmentation','resized','sortedBySize','regular','img'));
regular([1,2],:)=[];
big = dir(fullfile('D:','Studia','VI','POM','CNNsegmentation','resized','sortedBySize','biggg','img'));
big([1,2],:)=[];

%% Liczenie wsp Jaccarda z podzia³em na rozmiar - liczenie
clc;
cd('resized/sortedBySize/smol/')
for i=1:148
    toProcess = imread(fullfile('img', smol(i).name));
    result = semanticseg(toProcess,net);
    originalMask = imread(fullfile('mask', smol(i).name));
    
    binToCat = categorical(originalMask,[1,0],{'Cancer' 'Rest'});
    
    jacc=jaccard(result, binToCat);
    if isnan(jacc(1))
       jacc(1)=0; 
    elseif isnan(jacc(2))
       jacc(2)=0; 
    end
    
    similarity_small(i,1) = jacc(1);
    similarity_small(i,2) = jacc(2);
        
end
disp("MEAN VALUES OF JACCARD COEFFICIENT FOR SMALL OBJECTS:")
disp("Cancer:       "+sprintf('%.6f',mean(similarity_small(:,1))))
disp("Background:   "+sprintf('%.6f',mean(similarity_small(:,2))))

cd('D:\Studia\VI\POM\CNNsegmentation\')

cd('resized/sortedBySize/regular/')

for i=1:156
    toProcess = imread(fullfile('img', regular(i).name));
    result = semanticseg(toProcess,net);
    originalMask = imread(fullfile('mask', regular(i).name));
    
    binToCat = categorical(originalMask,[1,0],{'Cancer' 'Rest'});
    
    jacc=jaccard(result, binToCat);
    if isnan(jacc(1))
       jacc(1)=0; 
    elseif isnan(jacc(2))
       jacc(2)=0; 
    end
    
    similarity_regular(i,1) = jacc(1);
    similarity_regular(i,2) = jacc(2);
        
end
disp("MEAN VALUES OF JACCARD COEFFICIENT FOR MEDIUM OBJECTS:")
disp("Cancer:       "+sprintf('%.6f',mean(similarity_regular(:,1))))
disp("Background:   "+sprintf('%.6f',mean(similarity_regular(:,2))))

cd('D:\Studia\VI\POM\CNNsegmentation\')

cd('resized/sortedBySize/biggg/')

for i=1:158
    toProcess = imread(fullfile('img', big(i).name));
    result = semanticseg(toProcess,net);
    originalMask = imread(fullfile('mask', big(i).name));
    
    binToCat = categorical(originalMask,[1,0],{'Cancer' 'Rest'});
    
    jacc=jaccard(result, binToCat);
    if isnan(jacc(1))
       jacc(1)=0; 
    elseif isnan(jacc(2))
       jacc(2)=0; 
    end
    
    similarity_large(i,1) = jacc(1);
    similarity_large(i,2) = jacc(2);
        
end
disp("MEAN VALUES OF JACCARD COEFFICIENT FOR LARGE OBJECTS:")
disp("Cancer:       "+sprintf('%.6f',mean(similarity_large(:,1))))
disp("Background:   "+sprintf('%.6f',mean(similarity_large(:,2))))

cd('D:\Studia\VI\POM\CNNsegmentation\')















