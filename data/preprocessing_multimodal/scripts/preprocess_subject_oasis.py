import os 
import shutil 
import subprocess
import glob
import argparse
import random
import sys
import pandas as pd 

def download_oasis_scans(ids_list, output_dir): 

    df = pd.DataFrame({"image_id":ids_list})
    temp_df_path = os.path.join(output_dir, "temp_" + ids_list[0] + ".csv") 
    df.to_csv(temp_df_path, index=False)

    subprocess.run(["./download_oasis_scans.sh", temp_df_path, output_dir, "T1w"])

    os.remove(temp_df_path)

def preprocess_mri(id, master_folder):

    try:
        # Get mri folder path
        mri_folder = os.path.join(master_folder, id)

        subdir_names = []
        mri_filepath = ""

        for dirpath, dirnames, filenames in os.walk(mri_folder):

            # Save subdir names for later cleanup
            if dirpath == mri_folder:
                subdir_names = dirnames

            # Find first T1w scan 
            for file in filenames: 
                if "T1w" in file and ( file.endswith(".nii") or file.endswith(".nii.gz")):
                    mri_filepath = os.path.join(dirpath, file)
                    break

        # New file name 
        new_path = os.path.join(mri_folder, id + ".nii.gz")

        # Copy it and rename it 
        shutil.copyfile(mri_filepath, new_path)

        # Delete other folders 
        for subdir in subdir_names: 
            shutil.rmtree( os.path.join( mri_folder, subdir) )

        # Preprocess
        subprocess.run(["matlab", "-batch", f"preprocess_mri('{new_path}')"], check=True, env={
            "PATH": "/software/matlab-2023a-el8-x86_64/bin:/home/cestari/.conda/envs/adni_rcc/bin:/software/python-anaconda-2022.05-el8-x86_64/condabin:/home/cestari/.vscode-server/data/User/globalStorage/github.copilot-chat/debugCommand:/home/cestari/.vscode-server/data/User/globalStorage/github.copilot-chat/copilotCli:/home/cestari/.vscode-server/bin/94e8ae2b28cb5cc932b86e1070569c4463565c37/bin/remote-cli:/project/aereditato/cestari/fsl/share/fsl/bin:/project/aereditato/cestari/fsl/share/fsl/bin:/home/cestari/.local/bin:/home/cestari/bin:/software/bin:/software/slurm-current-el8-x86_64/bin:/software/modules/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/opt/thinlinc/bin:/home/cestari/.vscode-server/extensions/ms-python.debugpy-2025.18.0-linux-x64/bundled/scripts/noConfigScripts",
            "SPM_PATH": "/project/aereditato/cestari/spm/spm"})
        
        return mri_folder

    except Exception as e:
        print(f"MRI preprocessing failed for {id}: {repr(e)}. Exiting.")
        
        # Delete folder if it exists
        if os.path.exists(mri_folder):
            shutil.rmtree(mri_folder)

        return None

def preprocess_pet(pet_id, bias_corr_path, def_field_path, brain_mask_path, master_folder):

    try:

         # Get mri folder path
        pet_folder = os.path.join(master_folder, pet_id)

        subdir_names = []
        pet_filepath = ""

        for dirpath, dirnames, filenames in os.walk(pet_folder):

            # Save subdir names for later cleanup
            if dirpath == pet_folder:
                subdir_names = dirnames

            # Find first .nii file
            for file in filenames: 
                if ( file.endswith(".nii") or file.endswith(".nii.gz")):
                    pet_filepath = os.path.join(dirpath, file)
                    break

        # New file name 
        new_path = os.path.join(pet_folder, pet_id + ".nii.gz")

        # Copy it and rename it 
        shutil.copyfile(pet_filepath, new_path)

        # Delete other folders 
        for subdir in subdir_names: 
            print(f"Removing {subdir}")
            shutil.rmtree( os.path.join( pet_folder, subdir) )
        
        # Run matlab preprocessing script
        subprocess.run(["matlab", "-batch", f"preprocess_pet('{pet_folder}','{bias_corr_path}', '{def_field_path}', '{brain_mask_path}')"], check=True, env={
            "PATH": "/software/matlab-2023a-el8-x86_64/bin:/home/cestari/.conda/envs/adni_rcc/bin:/software/python-anaconda-2022.05-el8-x86_64/condabin:/home/cestari/.vscode-server/data/User/globalStorage/github.copilot-chat/debugCommand:/home/cestari/.vscode-server/data/User/globalStorage/github.copilot-chat/copilotCli:/home/cestari/.vscode-server/bin/94e8ae2b28cb5cc932b86e1070569c4463565c37/bin/remote-cli:/project/aereditato/cestari/fsl/share/fsl/bin:/project/aereditato/cestari/fsl/share/fsl/bin:/home/cestari/.local/bin:/home/cestari/bin:/software/bin:/software/slurm-current-el8-x86_64/bin:/software/modules/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/opt/thinlinc/bin:/home/cestari/.vscode-server/extensions/ms-python.debugpy-2025.18.0-linux-x64/bundled/scripts/noConfigScripts",
            "SPM_PATH": "/project/aereditato/cestari/spm/spm"})
        
        # Get name of produced .nii file
        nii = glob.glob(os.path.join(pet_folder, "*.nii"))
    
        # Rename it to clean_w_masked_rstatic_ID.nii
        filename = os.path.join(pet_folder, 'clean_w_masked_rstatic_' + pet_id + ".nii")
        os.replace(nii[0], filename)
        
    except Exception as e:
        print(f"PET preprocessing failed for {pet_id}: {repr(e)}. Skipping.")
        
        # Remove PET folder if it was created 
        if os.path.exists(pet_folder):
            shutil.rmtree(pet_folder)
        

def main(subject_index, csv_path, data_dir):

    # Set random seed 
    random.seed(42)

    # Read dataset csv
    df = pd.read_csv(csv_path)

    # Get subject id from subject index
    all_subjects = sorted(df["subject_id"].unique().tolist())
    subject_id = all_subjects[int(subject_index)]

    # Filter for scans belonging to the current patient
    df = df[df["subject_id"] == subject_id].copy()

    # Filter for MRIs and PETs
    df_mri = df[ df["group"].isin(["MRI-T1-3T", "MRI-T1-1.5T"]) ].copy()
    df_pet = df[ df["group"].apply( lambda s : s[:3] == "PET") ].copy()

    # If no MRI is available for this patient, return (PET needs MRI for preprocessing)
    if len(df_mri) == 0:
        return 

    # Keep track of mri_folders and ids for later use
    mri_info = []

    # Preprocess subject's MRIs first
    for index, row in df_mri.iterrows():

        mri_id = str(row["image_id"])
        mri_folder = preprocess_mri(mri_id, data_dir)

        if mri_folder is not None:
            mri_info.append((mri_id, mri_folder))
        

    # Preprocess subject's PETs
    for index, row in df_pet.iterrows():

        pet_id = str(row["image_id"])

        # Randomly pick one mri to use for the pet preprocessing.
        # They're all from the same patient, so it doesn't really matter which.
        mri_id, mri_folder = random.choice(mri_info)

        # Compose paths to bias corrected image, deformation field and brain mask
        bias_corr_path = os.path.join(mri_folder, 'm' + mri_id + '.nii')
        def_field_path = os.path.join(mri_folder, 'y_' + mri_id + '.nii')
        brain_mask_path = os.path.join(mri_folder, 'brain_mask_pet_m' + mri_id + '.nii')
        
        preprocess_pet(pet_id, bias_corr_path, def_field_path, brain_mask_path, data_dir)

    # Delete extra files in MRI folders to save space
    for mri_id, mri_folder in mri_info:

        # Delete extra files in MRI folder to save space
        try:
            bias_corr_path = os.path.join(mri_folder, 'm' + mri_id + '.nii')
            def_field_path = os.path.join(mri_folder, 'y_' + mri_id + '.nii')
            brain_mask_path = os.path.join(mri_folder, 'brain_mask_pet_m' + mri_id + '.nii')
            
            to_delete = [ bias_corr_path, def_field_path, brain_mask_path]
            for p in to_delete:
                os.remove(p)
                
        except:
            print(f"Failed to remove MRI temporary files.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('index', help='Index of the unique subject')
    parser.add_argument('csv_path', help='Path to csv file containing mri and pet ids.')
    parser.add_argument('data_dir', help='Path to data dir where preprocessed files will be stored')

    args = parser.parse_args(sys.argv[1:])

    print(args.index)
    print(args.csv_path)
    print(args.data_dir)

    main(args.index, args.csv_path, args.data_dir)
