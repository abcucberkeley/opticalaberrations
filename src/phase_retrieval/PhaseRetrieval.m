function [PR_Zcoefficients]=PhaseRetrieval(psf_path, dx, dz, lamd)

    %% Initial Parameters
    N_sampling = 257;
    NA = 1;
    index = 1.33;
    DM_FF = 1;
    DM_rotation = 180; %180
    DM_flip = 0; % 0 -> no DM flip, 1 -> vertical DM flip, 2 -> horizontal DM flip
    subsampleZ = 1; % works fine with 2 (faster), but more robust when set to 1
    residualThres = 10^-5; % convergence residual threshold to hit before stopping (10^-9 too much?)
    MaxIter = 25;
    flipZstack = false; % change to true if the scan direction is reversed or negative
    flipYstack = false; % change to true if the scan direction is reversed or negative
    flipXstack = false; % change to true if the scan direction is reversed or negative
    deskew = false;
    window_halfwidth = 8;

    %% Load and crop PSF
    PSF_measured = double(readtiff(psf_path));
    %PSF_measured = permute(PSF_measured, [2,1,3]);

    if flipYstack
        PSF_measured = flip(PSF_measured,1);
    end

    if flipXstack
        PSF_measured = flip(PSF_measured,2);
    end

    if flipZstack
        PSF_measured = flip(PSF_measured,3);
    end

    if deskew
        dz = sind(theta)*ds;
        PSF_measured = deskewFrame3D(PSF_measured,theta, ds, dx, 'reverse', true, 'Crop', true);
    end
    zAniso = dz/dx;

    [ny,nx,nz] = size(PSF_measured);
    [y,x,z] = ndgrid(1:ny,1:nx,1:nz);
    [Y,X,Z] = ndgrid(1:ny,1:nx,1:1/zAniso:nz);

    if deskew
        mask = logical(PSF_measured);
        PSF_measured(~mask) = 0;
    end

    % PSF_offset = thresholdOtsu(PSF_measured((PSF_measured(:)>minBk & PSF_measured(:)<prctile(PSF_measured(:),99.99))));
    PSF_offset = 1.5*mode(PSF_measured(PSF_measured>0));
    PSF_measured=PSF_measured-PSF_offset;
    PSF_measured(PSF_measured<0)=0;
    %idivide(size(PSF_measured, 1), int8(2));

    %Only look near the center of the ROI
    [ny,nx,nz]=size(PSF_measured);
    mask=zeros(ny,nx,nz);
    mask(max(1,ny/2-2*window_halfwidth):min(ny,ny/2+2*window_halfwidth),max(1,nx/2-2*window_halfwidth):min(nx,nx/2+2*window_halfwidth),max(1,nz/2-2*window_halfwidth):min(nz,nz/2+2*window_halfwidth))=1;
    PSF_measured=PSF_measured.*mask;

    [~,I] = max(PSF_measured(:));
    [PSF_center(1), PSF_center(2), PSF_center(3)] = ind2sub(size(PSF_measured), I);


    %Crop to region around PSF
    PSF_measured=PSF_measured(PSF_center(1)-window_halfwidth:PSF_center(1)+window_halfwidth,PSF_center(2)-window_halfwidth:PSF_center(2)+window_halfwidth,:);

    [nx,ny,nz]=size(PSF_measured);

    PSF = zeros(N_sampling,N_sampling,nz);
    PSF(round(N_sampling/2)-window_halfwidth:round(N_sampling/2)+window_halfwidth,round(N_sampling/2)-window_halfwidth:round(N_sampling/2)+window_halfwidth,:) = PSF_measured;
    PSF = sqrt(PSF);

    [C,I] = max(PSF(:));
    [PSF_center(1), PSF_center(2), PSF_center(3)] = ind2sub(size(PSF), I);


    %% Pupil Plane
    dk=2*pi/N_sampling/dx;
    kx=[-(N_sampling-1)/2:1:(N_sampling-1)/2]*dk;
    ky=kx;
    [kxx, kyy]=meshgrid(kx,ky);
    kr=sqrt(kxx.^2+kyy.^2);

    pupil_mask = gather(kr<NA*(2*pi/lamd));
    pupil_field = pupil_mask;

    Defocus = gather(zeros(N_sampling,N_sampling,nz));
    for j = 1:nz
        Defocus(:,:,j) = exp(1i*sqrt(((2*index*pi/lamd)^2 - kr.^2).*((2*index*pi/lamd)^2 > kr.^2))*-dz*(j-PSF_center(3)));
    end


    %% Pupil displacement

    gPSF = gather(PSF);

    mask = pupil_field;
    d = ones(size(pupil_field));

    nn = 0;
    while max(real(d(mask))) > residualThres || max(imag(d(mask))) > residualThres

        PSFA = gather(zeros(N_sampling,N_sampling,nz));
        temp = gather(zeros(N_sampling,N_sampling));
        prev_pupil_field = pupil_field;

        for j = 1:subsampleZ:nz
            PSFA(:,:,j) = gPSF(:,:,j).*exp(1i*angle(ifftshift(fft2(fftshift(pupil_field.*Defocus(:,:,j))))));
            temp = temp + ifftshift(ifft2(fftshift(PSFA(:,:,j))))./Defocus(:,:,j);
        end

        pupil_field = (temp/nz).*pupil_mask;
        d = pupil_field-prev_pupil_field;
        nn = nn+1;
        if nn == MaxIter
            break
        end
    end

    pupil_mask = gather(pupil_mask);
    pupil_field = gather(pupil_field);
    %pupil_phase = GoldsteinUnwrap2D_r1(pupil_mask.*exp(1i*angle(pupil_field)),pupil_mask);
    pupil_phase = unwrap2D(angle(pupil_field),pupil_mask.*255, size(pupil_mask,2), size(pupil_mask,1), 0, 0, 0, 0);
    pupil_phase(isnan(pupil_phase)) = 0;


    if DM_flip == 1 || DM_flip == 2
        pupil_phase = flip(pupil_phase, DM_flip);
    end

    pupil_displacement = (pupil_phase/2/pi*lamd)'; %Convert pupil phase into wavelengths'
    pupil_displacement = pupil_displacement.*pupil_mask; %remask after unwrapping
    pupil_displacement = imrotate(pupil_displacement,DM_rotation);

    x_pupil = kxx(:)/(NA*(2*pi/lamd))*DM_FF; y_pupil = kyy(:)/(NA*(2*pi/lamd))*DM_FF;
    [theta_pupil,r_pupil] = cart2pol(x_pupil,y_pupil);
    is_in_circle = ( r_pupil <= 1 );

    N = []; M = [];
    for n = 0:9
        N = [N n*ones(1,n+1)];
        M = [M -n:2:n];
    end

    Z = zernfun(N,M,r_pupil(is_in_circle),theta_pupil(is_in_circle));
    PR_Zcoefficients = Z\pupil_displacement(is_in_circle);
    PR_Zcoefficients(1:3)=0;
    PR_Zcoefficients(5)=0;
