addpath(genpath('~/fiona/ABCcode/XR_GU_Repository'));
addpath(genpath('~/fiona/ABCcode/XR_Repository'));

%% Step 1: load data
%  rt is the root directory of data, you may provide root directory. If it
%  is not provided, the load condition data function will prompt out to
%  choose the root directory interactively.
rt = '~/nvme/thayer/covid/exp1_corp/';
% follow the below when asking for inputs (They may be different for your own experiments):
% Enter the number of channels: 2
% click folder name 'ch1' when it first prompts out.
% click folder name 'ch2' when it prompts out again.
% Enter the fluorescent marker for channel 1: gfp
% Enter the fluorescent marker for channel 2: rfp
data = XR_loadConditionData3D(rt, 'MovieSelector', 'exp', 'dz', 0.5, 'PixelSize',0.108, 'Angle', 32.45);

%% Step 2: Estimate psf sigmas if there are calibration files and they are not available (optional)
% The sigmas of psfs are estimated separately. The filename is provided as
% input for the estimation.
ch1_psf_filename = '~/nvme/thayer/covid/PSF/DS/488nm_CamB_ch0_hexrect_NA0p25_dNAa0p08_500nmstep.tif';
[sigmaXY_ch1, sigmaZ_ch1] = GU_estimateSigma3D(ch1_psf_filename, []);
% ch2_psf_filename = '';
% [sigmaXY_ch2, sigmaZ_ch2] = GU_estimateSigma3D(ch2_psf_filename, []);
% Because in calibration (of this data), the calibration image is
% approximately isotropic, which is is not true for the data. We need to
% divide the sigma in z-axis by the Anisotropic factor in this axis.
sigma_mat = [sigmaXY_ch1, sigmaZ_ch1 ./ data(1).zAniso];
sigma_mat = [sigmaXY_ch1, sigmaZ_ch1];
%% Step 3: deskew, and correct XZ offsets between channels
% name of directory that stores the results (under the primary channel directory).
aname = 'results';
% deskew movies to place z-slices in their correct relative
% physical poisitions. The results are stored in 'DS' directory under the
% channel directory.
% data = deskewData(data, 'Overwrite', true, 'SkewAngle', 32.5, 'aname', aname, ...
    % 'Rotate', false,'sCMOSCameraFlip', false, 'Crop', false);
% In some cases, there are some offsets between different channels, due to
% camera/microscope, especially in x and z. Here we use maximum
% cross-correlation to correct them. If there is correction, the DS results
% are overwritten.
% XR_correctXZoffsetData3D(data, 'Overwrite', true);
%% Step 4: Detection
% Detect diffraction-limited points using deskewed data.
apath = arrayfun(@(i) [i.source filesep aname filesep], data, 'unif', 0);
tic
XR_runDetection3D(data, 'Sigma', sigma_mat, 'Overwrite', true,...
    'WindowSize', [], 'ResultsPath', apath,'Mode', 'xyzAsrc',...
    'FitMixtures', false, 'MaxMixtures', 1, ...
    'DetectionMethod', 'speedup', 'BackgroundCorrection', true);
toc