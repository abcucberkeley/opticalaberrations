function []=TA_Decon(img, psf, iters, savepath)
    img = readtiff(img);
    psf = readtiff(psf);
    im = decon_lucy_function(img, psf, iters);
    writetiff(im, savepath);