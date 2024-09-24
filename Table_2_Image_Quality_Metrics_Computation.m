%% Read the images (to change between 10-1, 10-2 & 10-3)
ref = imread('10-1E.png');
A = imread('10-1GO.png');
B = imread('10-1GU.png');

%% Calculate PSNR/SNR
[peaksnr, snr] = psnr(A, ref);
[peaksnr2, snr2] = psnr(B, ref);
fprintf('\n The O-net PSNR value is %0.4f', peaksnr);
fprintf('\n The O-net SNR value is %0.4f \n', snr);
fprintf('\n The U-net PSNR value is %0.4f', peaksnr2);
fprintf('\n The U-net SNR value is %0.4f \n', snr2);

%% Calculate IMSE (https://www.mathworks.com/help/images/ref/immse.html)
err = immse(A, ref);
err2 = immse(B, ref);
fprintf('\n The O-Net mean-squared error (IMSE) is %0.4f\n', err);
fprintf('\n The U-Net mean-squared error (IMSE) is %0.4f\n', err2);

%% Calculate SSIM (https://www.mathworks.com/help/images/ref/ssim.html)
[ssimval,ssimmap] = ssim(A,ref);
[ssimval2,ssimmap2] = ssim(B,ref);
ssimscore = squeeze(ssimval);
ssimscore2 = squeeze(ssimval2);
fprintf('\n The O-Net SSIM is %0.4f\n', ssimscore);
fprintf('\n The U-Net SSIM is %0.4f\n', ssimscore2);