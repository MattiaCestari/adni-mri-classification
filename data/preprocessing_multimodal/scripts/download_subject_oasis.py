import pandas as pd 
import os
import sys
import subprocess 
import argparse

def download_oasis_scans(ids_list, output_dir): 

    df = pd.DataFrame({"image_id":ids_list})
    temp_df_path = os.path.join(output_dir, "temp_" + ids_list[0] + ".csv") 
    df.to_csv(temp_df_path, index=False)

    subprocess.run(["./download_oasis_scans.sh", temp_df_path, output_dir, "T1w"])

    os.remove(temp_df_path)

def main(index_start, index_end, csv_path, data_dir):

    df = pd.read_csv(csv_path)
    subjs = sorted(list(df["subject_id"].unique()))[index_start:index_end+1]

    scans_ids = list(df[ df["subject_id"].isin(subjs)]["image_id"])

    download_oasis_scans(scans_ids, data_dir)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('index_start', help='Index of the first unique subject to download')
    parser.add_argument('index_end', help='Index of the last unique subject to download')
    parser.add_argument('csv_path', help='Path to csv file containing mri and pet ids.')
    parser.add_argument('data_dir', help='Path to data dir where preprocessed files will be stored')

    args = parser.parse_args(sys.argv[1:])

    main(int(args.index_start), int(args.index_end), args.csv_path, args.data_dir)