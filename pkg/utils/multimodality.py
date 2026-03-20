import pandas as pd

from pysat.examples.rc2 import RC2
from pysat.formula import WCNF
from tqdm import tqdm
import numpy as np

def create_multimodal_samples(scans, tolerance):
        """
        Creates disjoint multimodal samples, one for every
        scan of the minority modality. 

        Returns (samples, used_ids)
        """

        # Helper functions
        def check_valid(sample, tolerance):
            """
            Checks if scans in sample are within tolerance. 
            """
            
            # Extract scans dates
            dates = [ s[1] for s in sample if s is not None]
            
            # Check wheter timeframe is below tolerance
            return abs( max(dates) - min(dates)).days <= tolerance

        def weight(sample):
            """
            Returns weight of sample for MaxSAT. 
            """
            
            # Weight = number of modalities
            return len([ scan for scan in sample if scan is not None])

        def disjoint(s1, s2):
            """
            Returns whether sample 1 and sample 2 are disjoint
            """
            id1 = { e[0] for e in s1 if e is not None }
            id2 = { e[0] for e in s2 if e is not None }

            return id1.isdisjoint(id2)

        if scans.empty:
            return ([], [])

        # Ensure right data type 
        scans["image_date"] = pd.to_datetime(scans["image_date"])

        # Get lowest count modality
        min_mode = scans.groupby("modality")["image_id"].nunique().idxmin()
        
        # Get other available modalities
        avail_modes = sorted(scans["modality"].unique())
        avail_modes.remove(min_mode)

        # Store candidate multimodal samples
        candidate_samples = []
        
        # Use lowest count modality as anchor modality
        for _, row in scans[scans["modality"] == min_mode].iterrows():
                
            # Get scans within tolerance 
            tol_scans = scans[(scans["modality"] != min_mode) 
                            & (abs(scans["image_date"] - row["image_date"]).dt.days <= tolerance )]
            
        
            # For each modality, grow possible samples under tolerance
            samples = [ (tuple(row[["image_id", "image_date"]]),) ]

            for mode in avail_modes:
                    
                # Skip anchor modality
                if mode == min_mode:
                    continue
                
                # Scans to choose from this modality (or choose none)
                options = [ e for e in tol_scans[ tol_scans["modality"] == mode][["image_id", "image_date"]].itertuples(name=None, index=False) ] + [None]
                
                # Grow samples with options from current modality  
                new_samples = []
                
                for s in samples:
                    for o in options:
                        # Create new multimodal sample
                        new = s + (o,)
                        
                        # If under tolerance, add
                        if check_valid(new, tolerance):
                            new_samples.append(new)

                samples = new_samples
        
            # Add samples 
            candidate_samples += samples
            
        if not candidate_samples:
            return ([], [])
        
        # Solve ILP problem to find best subset of disjoint multimodal samples
        # Create SAT solver instance
        wcnf = WCNF()
            
        # Add soft clauses with weight
        for i, s in enumerate(candidate_samples):
            wcnf.append([i+1], weight=weight(s))
                
        # Add disjoint constraints
        for i in range(len(candidate_samples)):
            for j in range(i+1, len(candidate_samples)):
                        
                # If samples are not disjoint, add clause
                if not disjoint(candidate_samples[i], candidate_samples[j]):
                    wcnf.append([-(i+1), -(j+1)])
                            
        # Solve
        solver = RC2(wcnf)
        model = solver.compute()

        # Get selected samples
        selected_samples = [ s for i,s in enumerate(candidate_samples) if (i+1) in model]

        all_modes = [min_mode] + avail_modes

        # return list of samples
        ret = []

        # And of the scan ids used in these samples
        used_ids = []

        # Convert to dicts for ease
        for s in selected_samples:

            # Update used ids
            for e in s:
                if e is not None:
                    used_ids.append(e[0])

            # Convert sample to dict
            s = {all_modes[i]:s[i][0] for i in range(len(all_modes)) if s[i] is not None}
            
            ret.append(s)

        return (ret, used_ids)


def create_multimodal_dataframe(df_scans, tolerance=180, passes=2, extra_vars=["age", "gender", "MMSE", "CDR"]):

        # Entries for mmodal dataframe
        entries = []

        # All modalities
        modalities = sorted(df_scans["modality"].unique())

        for group, indices in df_scans.groupby(["subject_id", "diagnosis"]).groups.items():
            
            # Get scans of current group
            scans = df_scans.loc[indices]
            
            samples = []

            for i in range(passes):

                current_samples, used_ids = create_multimodal_samples(scans, tolerance)

                # Append samples 
                samples += current_samples

                # Update scans 
                scans = scans[~scans["image_id"].isin(used_ids)]

            # Add multimodal samples to final dataframe entries
            for s in samples:

                s["diagnosis"] = group[1]
                s["subject_id"] = group[0]
                
                entries.append(s)

            # Add remaining scans as unimodal samples to final dataframe entries
            for _, row in scans.iterrows():

                s = {row["modality"]:row["image_id"]}
                s["diagnosis"] = group[1]
                s["subject_id"] = group[0]

                entries.append(s)

        # Create final dataframe
        df_final = pd.DataFrame(entries)

        # Add stratification keys: diagnosis + presence of scan belonging to different groups
        # (MRI 3T, MRI 1.5T, PET-FDG, PET-AV45, PET-AV1451, PET-FBB, PET-MK6240, PET-NAV4694, PET-PI2620)
        keys = []

        # Add extra variables as well
        to_concat = []

        for index, row in df_final.iterrows():

            # Initalize key as diagnosis
            key = str(int(row["diagnosis"]))

            # Get scans list
            scans = []
            matched_rows = []

            for mode in modalities:
                if mode in row.index:
                    scan_id = row[mode]
                    scans.append(scan_id)

                    if not pd.isna(scan_id):
                        matched_rows.append(
                            df_scans.loc[df_scans["image_id"] == scan_id, extra_vars].iloc[0]
                        )
                else:
                    scans.append(np.nan)

            # Build one combined row of extra vars
            if matched_rows:
                matched_df = pd.DataFrame(matched_rows)

                extra_values = {}
                for col in extra_vars:
                    if pd.api.types.is_numeric_dtype(matched_df[col]):
                        extra_values[col] = matched_df[col].mean()
                    else:
                        # for categorical vars like gender, take first non-null value
                        non_null = matched_df[col].dropna()
                        extra_values[col] = non_null.iloc[0] if not non_null.empty else np.nan
            else:
                extra_values = {col: np.nan for col in extra_vars}

            to_concat.append(extra_values)

            # For each group, check if there's a scan belonging to that group
            for group in sorted(df_scans["group"].unique()):
                if group in df_scans[df_scans["image_id"].isin(scans)]["group"].tolist():
                    key += "1"
                else:
                    key += "0"

            keys.append(key)

        df_final[extra_vars] = pd.DataFrame(to_concat, index=df_final.index)
        df_final["strat_key"] = keys

        return df_final