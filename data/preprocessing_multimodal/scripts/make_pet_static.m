function static_pet = make_pet_static(nii_dir)
% MAKE_PET_STATIC  Merge all 3D NIfTI volumes in nii_dir into 4D, then average to static PET.
% Output filenames inherit the original PET basename with prefixes.

    nii_dir = char(nii_dir);

    % SPM setup
    addpath(genpath(char(getenv('SPM_PATH'))));
    spm('Defaults','PET');
    spm_jobman('initcfg');

    % List .nii.gz files
    files = dir(fullfile(nii_dir, '*.nii.gz'));
    if ~isempty(files)

        % Convert to .nii
        for i = 1:length(files)
            
            gz_path = fullfile(files(i).folder, files(i).name);
            outdir = files(i).folder;

            gunzip(gz_path, outdir);

            % Optional: name of the unzipped file
            nii_file = files(i).name(1:end-3);

        end
    end

    files = dir(fullfile(nii_dir, '*.nii'));
    if isempty(files)
        fprintf('No .nii files found in directory: %s', nii_dir);
    end

    % ---- derive base name from first frame ----
    [~, base, ext] = fileparts(files(1).name);
    if strcmp(ext, '.gz')
        [~, base] = fileparts(base); % handles .nii.gz if any slipped through
    end

    % Build N×M char matrix: one file per row (SPM convention)
    file_list = char(fullfile(nii_dir, {files.name})');

    % ---- merged 4D filename ----
    merged_nii = fullfile(nii_dir, ['merged_' base '_4D.nii']);
    spm_file_merge(file_list, merged_nii);

    % Load 4D header list
    V = spm_vol(merged_nii);

    % If already single volume, just return it
    if numel(V) == 1
        static_pet = merged_nii;
        return;
    end

    % Accumulate mean
    Ysum = [];
    for i = 1:numel(V)
        Yi = spm_read_vols(V(i));
        if isempty(Ysum)
            Ysum = zeros(size(Yi), 'like', Yi);
        end
        Ysum = Ysum + Yi;
    end
    Ymean = Ysum / numel(V);

    % ---- static PET filename ----
    static_pet = fullfile(nii_dir, ['static_' base '.nii']);

    Vout = V(1);
    Vout.fname = static_pet;
    Vout.dt    = [16 0];   % float32
    Vout.n     = [1 1];

    spm_write_vol(Vout, Ymean);

    fprintf('Static PET written to:\n  %s\n', static_pet);
end
