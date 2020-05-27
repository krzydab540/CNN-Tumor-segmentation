masks = dir(fullfile('D:','Studia','VI','POM','CNNsegmentation','resized','augs','masks'));
origs = dir(fullfile('D:','Studia','VI','POM','CNNsegmentation','resized','augs','imgs'));

masks([1,2],:)=[];
origs([1,2],:)=[];

NF = 462;
percentage = zeros(NF,3); % ile bia³ych, ile czarnych, procent wype³nienia
counter = [0 0 0]; %licznik do numerowania - ma³e œrednie du¿e

cd('resized/augs')

for k = 1 : NF
  mask = imread(fullfile('masks', masks(k).name));
  image = imread(fullfile('imgs', origs(k).name));
  
  %liczenie iloœci pikseli
  mask = im2bw(mask);
  noPixels = sum(mask(:));
  noRest = 16000-noPixels;  
  
  %zapisanie wyników i % wype³nienia
  percentage(k,1) = noPixels;
  percentage(k,2) = noRest;
  percentage(k,3) = (noPixels/16000)*100;
    
  %zapisanie do odpowiedniego folderu
  if(percentage(k,3)<0.1875)
      counter(1)= counter(1)+1;
      imwrite(image, strcat("D:\Studia\VI\POM\CNNsegmentation\resized\sortedBySize\smol\img\o_", int2str(counter(1)), ".png"));
      imwrite(mask, strcat("D:\Studia\VI\POM\CNNsegmentation\resized\sortedBySize\smol\mask\o_", int2str(counter(1)), ".png"));

  elseif(percentage(k,3)>0.1875 && percentage(k,3)<1)
      counter(2)= counter(2)+1;
      imwrite(image, strcat("D:\Studia\VI\POM\CNNsegmentation\resized\sortedBySize\regular\img\o_", int2str(counter(2)), ".png"));
      imwrite(mask, strcat("D:\Studia\VI\POM\CNNsegmentation\resized\sortedBySize\regular\mask\o_", int2str(counter(2)), ".png"));
  else
      counter(3)= counter(3)+1;
      imwrite(image, strcat("D:\Studia\VI\POM\CNNsegmentation\resized\sortedBySize\biggg\img\o_", int2str(counter(3)), ".png"));
      imwrite(mask, strcat("D:\Studia\VI\POM\CNNsegmentation\resized\sortedBySize\biggg\mask\o_", int2str(counter(3)), ".png"));
  end
  
  disp(counter)

end

cd('D:\Studia\VI\POM\CNNsegmentation\')

