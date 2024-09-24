% Read the images
pref = imread("6-1E-1.png");
UPCM = imread("6-1GU-1.png");
OPCM = imread("6-1GO-1.png");
dref = imread("6-1E-2.png");
UDIC = imread("6-1GU-2.png");
ODIC = imread("6-1GO-2.png");

%% Find the SSIM (global) & SSIM (local) for U-Net generated DIC image
[dssimval,dssimmap] = ssim(UDIC,dref);
imshow(dssimmap,[])
title("Local U-Net SSIM Map with Global SSIM Value: "+num2str(dssimval))

%% Find the SSIM (global) & SSIM (local) for O-Net generated DIC image
[dssimvala,dssimmapa] = ssim(ODIC,dref);
imshow(dssimmapa,[])
title("Local O-Net SSIM Map with Global SSIM Value: "+num2str(dssimvala))

%% Find the SSIM (global) & SSIM (local) for U-Net generated PCM image
[pssimval,pssimmap] = ssim(UPCM,pref);
imshow(pssimmap,[])
title("Local U-Net SSIM Map with Global SSIM Value: "+num2str(pssimval))

%% Find the SSIM (global) & SSIM (local) for O-Net generated PCM image
[pssimvala,pssimmapa] = ssim(OPCM,pref);
imshow(pssimmapa,[])
title("Local O-Net SSIM Map with Global SSIM Value: "+num2str(pssimvala))