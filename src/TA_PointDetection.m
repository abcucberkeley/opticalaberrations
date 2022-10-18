function []=TA_PointDetection(img, psf, dxy, dz, angle, sigma_xy, sigma_z)
    aname = 'results';
    data = struct();
    rt = fileparts(img);
    data.source = rt;
    data.channels = {rt};
    data.pixelSize = dxy;
    data.dz = dz;
    data.angle = angle;
    data.zAniso = dz*sind(angle)/dxy;
    data.framePathsDS = {{img}};
    data.movieLength = 1;
    data.date = [];
    data.framerate = NaN;
    data.markers = {488e-9};

    %% Estimate psf sigmas if there are calibration files and they are not available (optional)
    % The sigmas of psfs are estimated separately. The filename is provided as
    % input for the estimation.
    % [sigma_xy, sigma_z] = GU_estimateSigma3D(psf, []);
    sigma_mat = [sigma_xy, sigma_z];

    %% Detection
    % Detect diffraction-limited points using deskewed data.
    apath = arrayfun(@(i) [i.source filesep aname filesep], data, 'unif', 0);

    XR_runDetection3D(data, 'Sigma', sigma_mat, 'Overwrite', true,...
        'WindowSize', [], 'ResultsPath', apath,'Mode', 'xyzAc',...
        'FitMixtures', false, 'MaxMixtures', 1, ...
        'DetectionMethod', 'speedup', 'BackgroundCorrection', true);
