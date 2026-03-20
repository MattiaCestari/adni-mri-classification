function preprocess_mri(mri_path)

    % SPM setup
    addpath(genpath(char(getenv('SPM_PATH'))));
    spm('Defaults','FMRI');
    spm_jobman('initcfg');

    % --- Segmentation
    segment_outputs = segment(mri_path);

    % --- Create brain mask for MRI in native space
    c1 = spm_read_vols(spm_vol(segment_outputs.c1));
    c2 = spm_read_vols(spm_vol(segment_outputs.c2));
    brain_mask_mri = (c1 + c2) > 0.3;

    % --- Create brain mask for PET in native space
    c3 = spm_read_vols(spm_vol(segment_outputs.c3));
    brain_mask_pet = (c1 + c2 + c3) > 0.5;

    % --- Load bias-corrected MRI and apply mask
    bc_mri_vol = spm_vol(segment_outputs.m);
    mri = spm_read_vols(bc_mri_vol);
    mri(~brain_mask_mri) = 0;

    % --- Save masked image (native space)
    masked_mri_path = spm_file(bc_mri_vol.fname, 'prefix', 'masked_');
    bc_mri_vol.fname = masked_mri_path;
    spm_write_vol(bc_mri_vol, mri);

    % --- Save mri brain mask and normalize it
    mask_vol = bc_mri_vol;  % same grid/affine as MRI
    brain_mask_path = spm_file(segment_outputs.m, 'prefix', 'brain_mask_mri_');
    mask_vol.fname = brain_mask_path;
    mask_vol.dt    = [2 0]; % uint8
    spm_write_vol(mask_vol, uint8(brain_mask_mri));

    w_mri_mask_path = normalize(brain_mask_path, segment_outputs.y, 0);

    % --- Save pet brain mask in native space (DO NOT WARP IT)
    mask_vol = bc_mri_vol; % same grid/affine as MRI
    pet_mask_path = spm_file(segment_outputs.m, 'prefix', 'brain_mask_pet_');
    mask_vol.fname = pet_mask_path;
    mask_vol.dt    = [2 0];
    spm_write_vol(mask_vol, uint8(brain_mask_pet));  % <-- added missing semicolon

    % --- Normalize masked MRI to MNI space (interp default=4)
    w_img_path = normalize(masked_mri_path, segment_outputs.y);

    % Load warped mask + image
    w_mri_mask = spm_read_vols(spm_vol(w_mri_mask_path)) > 0.5;
    w_img      = spm_read_vols(spm_vol(w_img_path));

    % Sanity check
    if ~isequal(size(w_img), size(w_mri_mask))
        error('Size mismatch: w_img=%s, w_mask=%s', mat2str(size(w_img)), mat2str(size(w_mri_mask)));
    end

    % Clean artifacts
    w_img(isnan(w_img)) = 0;
    w_img(~w_mri_mask)  = 0;
    w_img(w_img < 0)    = 0;

    % 0-1 normalization
    w_img_max = max(w_img(:));
    w_img = w_img ./ (w_img_max + eps);

    % Save final clean image next to normalized image
    folder = fileparts(w_img_path);
    [~, name, ext] = fileparts(w_img_path);  % name includes the w_ prefix
    clean_path = fullfile(folder, ['clean_' name ext]);

    w_vol = spm_vol(w_img_path); % header from normalized image
    w_vol.fname = clean_path;
    spm_write_vol(w_vol, w_img);

    % ------------------------------------------------------------------
    % DELETE ALL OTHER WORKING FILES: keep only
    %   1) bias-corrected MRI (segment_outputs.m)
    %   2) deformation field (segment_outputs.y)
    %   3) final clean warped MRI (clean_path)
    %   4) PET mask
    % ------------------------------------------------------------------
    keep = { segment_outputs.m, segment_outputs.y, clean_path, pet_mask_path};

    d = dir(folder);
    for i = 1:numel(d)
        if d(i).isdir, continue; end
        fp = fullfile(folder, d(i).name);
        if any(strcmp(fp, keep))
            continue;
        end
        try
            delete(fp);
        catch
            % ignore delete failures (e.g., file locked); optional: fprintf warning
        end
    end

    fprintf('Finished preprocessing: %s\n', mri_path);
end
