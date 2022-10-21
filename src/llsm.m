addpath(genpath('~/fiona/ABCcode/XR_Repository'));

%% parameters
rt = '~/Gitlab/opticalaberrations/lattice/';
b = load([rt, 'plane wave sets for common lattices 1-7-22.mat']);
lattice_name = 'fish'
algorithm = 'original';
xyPol = [1, 1, 0];
switch lattice_name
    case 'covid'
        lattice_descrip = 'HexRect';
        PW =  b.PW_HexRect;
        algorithm = 'new';
        NAlattice = .25;
        NAsigma = .08;
        NAannulus = [0.6, 0.2];
        crop_factor = 0.1;
    case 'fish'
        lattice_descrip = 'YuMB';
        PW =  b.PW_Sq45;
        algorithm = 'new';
        NAlattice = .35;
        NAsigma = .1;
        NAannulus = [0.4, 0.3];
        crop_factor = 0.1;
end
NAdet = 1.0;
xz_det_PSF = b.xz_PSF_RW_510nm_NA1p0;
xz_det_OTF = b.xz_OTF_RW_510nm_NA1p0;
gamma = .5;
yplot_range = 99;
ystepsize_in = 3;
visualize =~false;
pixsize = 0.1;
imgpath = [rt, 'simulations/'];
mkdir(imgpath);
subfolder = ['NAlattice', num2str(NAlattice, '%1.2f'), filesep, lattice_descrip, filesep, 'NAAnnulusMax', num2str(NAannulus(1), '%1.2f'), filesep, 'NAsigma', num2str(NAsigma, '%1.2f'), filesep];
resultPath = sprintf('%s/%s/decon_simulation/', imgpath, subfolder);
mkdir(resultPath);
llsFn = sprintf('%s/PSF_OTF_simulation.mat', resultPath);
PSF_folder = sprintf('%sPSFs/', resultPath);
mkdir(PSF_folder);
figurePath = sprintf('%sfigures/', resultPath);
mkdir(figurePath);

%% run simulation
tic
[yFWHM, PWb, OnAxisIntensity, CumulativeFWHM, period, OverallDitheredPSF, Overall_Dithered_xz_OTF2, DitheredxzPSFCrossSection, ...
    PlotSLMPhase, MaskIntensity, SampleIntensityAtFocus, DitheredIntensity, OverallCrossSection, SheetCrossSection, DetPSFCrossSection] ...
    = Calc_and_Plot_3D_LLS_PSFs_and_OTFs_v9e_XR(algorithm, lattice_descrip, xyPol, PW, NAlattice, NAsigma, NAannulus, crop_factor, ...
    NAdet, xz_det_PSF, xz_det_OTF, gamma, yplot_range, ystepsize_in, imgpath, subfolder, visualize);
toc
save('-v7.3', llsFn, 'yFWHM', 'PWb', 'OnAxisIntensity', 'CumulativeFWHM', ...
    'OverallDitheredPSF', 'Overall_Dithered_xz_OTF2', 'DitheredxzPSFCrossSection', ...
    'PlotSLMPhase', 'MaskIntensity', 'SampleIntensityAtFocus', 'DitheredIntensity', ...
    'OverallCrossSection', 'SheetCrossSection', 'DetPSFCrossSection', 'yplot_range', ...
    'ystepsize_in');