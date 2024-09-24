% Read the images
pref = imread("10-2E-1.png");
UPCM = imread("10-2GU-1.png");
OPCM = imread("10-2GO-1.png");
dref = imread("10-2E-2.png");
UDIC = imread("10-2GU-2.png");
ODIC = imread("10-2GO-2.png");

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