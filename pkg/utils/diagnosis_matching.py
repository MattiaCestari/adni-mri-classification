import pandas as pd


def match_mental_scores(df_scan, 
                df_visit, 
                tolerance, 
                column=["diagnosis", "DIAGNOSIS"], 
                date_col=["exam_date","EXAMDATE"]):
    """

    """

    to_concat = []

    df_scan = df_scan.copy()
    df_visits = df_visits.copy()

    # Rename columns 
    df_visits = df_visits.rename(columns={column[1]:column[0], date_col[1]:date_col[0]})
 
    date_col = date_col[0]

    # Ensure datetime dtype once
    df_visits[date_col] = pd.to_datetime(df_visits[date_col])
    df_scan[date_col] = pd.to_datetime(df_scan[date_col])
    
    # Keep track of original cols to preserve (+ diagnosis and exam date)
    cols_to_keep = df_scan.columns.tolist() + ["diagnosis", "exam_date"]

    pass 

def match_diagnosis(df_scan, df_diagnostic, tolerance):
        """
        Adds diagnosis column to scan dataframe by matching it with temporally closest visit
        in the diagnositc dataframe. 
        """

        to_concat = []
    
        df_scan = df_scan.copy()
        df_diagnostic = df_diagnostic.copy()

        # Rename columns 
        df_diagnostic = df_diagnostic.rename(columns={"EXAMDATE":"exam_date", "DIAGNOSIS":"diagnosis"})

        # Ensure datetime dtype once
        df_diagnostic["exam_date"] = pd.to_datetime(df_diagnostic["exam_date"])
        df_scan["image_date"] = pd.to_datetime(df_scan["image_date"])

        # Keep track of original cols to preserve (+ diagnosis and exam date)
        cols_to_keep = df_scan.columns.tolist() + ["diagnosis", "exam_date"]

        for subj in df_scan["subject_id"].unique():

            # Get subject visits and scans
            subj_vis = df_diagnostic[df_diagnostic["PTID"] == subj].copy()
            subj_scans = df_scan[df_scan["subject_id"] == subj].copy()

            # Drop NaN values
            subj_vis = subj_vis.dropna(subset=["exam_date"])
            subj_scans = subj_scans.dropna(subset=["image_date"])
            
            # Merge_asof requires sorting by the key
            subj_vis = subj_vis.sort_values("exam_date")
            subj_scans = subj_scans.sort_values("image_date")

            # Match nearest withing tolerance
            nearest = pd.merge_asof(
                subj_scans,
                subj_vis[["exam_date", "diagnosis"]],
                left_on="image_date",
                right_on="exam_date",
                direction="nearest",
                tolerance=pd.Timedelta(days=tolerance),
            )

            # Get matched and append to list
            matched = nearest[nearest["diagnosis"].notna()].copy()

            matched = matched[ cols_to_keep ]  
    
            to_concat.append(matched)

            # Matched ids
            matched_ids = set(matched["image_id"])

            # If all scans matched, continue
            if len(matched_ids) == len(subj_scans):
                continue

            # Remaining scans (no visit within tolerance)
            remaining = subj_scans[~subj_scans["image_id"].isin(matched_ids)].copy()

            # Get closest visits before
            before = pd.merge_asof(
                remaining,
                subj_vis[["exam_date", "diagnosis"]],
                left_on="image_date",
                right_on="exam_date",
                direction="backward",   # visit on/before scan
                allow_exact_matches=True,
            ).rename(columns={"exam_date": "exam_date_before", "diagnosis": "diagnosis_before"})

            # Get closest visits after
            after = pd.merge_asof(
                remaining,
                subj_vis[["exam_date", "diagnosis"]],
                left_on="image_date",
                right_on="exam_date",
                direction="forward",    # visit on/after scan
                allow_exact_matches=True,
            ).rename(columns={"exam_date": "exam_date_after", "diagnosis": "diagnosis_after"})

            
            # Bracket: closest visit before + closest visit after!
            bracket = before.merge(
                after[["image_id", "exam_date_after", "diagnosis_after"]],
                on="image_id",
                how="inner",
            )

            # need both sides, and diagnosis must agree
            bracket = bracket[
                bracket["diagnosis_before"].notna()
                & bracket["diagnosis_after"].notna()
                & (bracket["diagnosis_before"] == bracket["diagnosis_after"])
            ].copy()

            bracket["diagnosis"] = bracket["diagnosis_after"]

            # Keep closest exam date
            d_before = (bracket["image_date"] - bracket["exam_date_before"]).abs()
            d_after  = (bracket["exam_date_after"] - bracket["image_date"]).abs()
            bracket["exam_date"] = bracket["exam_date_before"].where(d_before <= d_after, bracket["exam_date_after"])

            # Keep only relevant columns
            bracket = bracket[cols_to_keep]
    
            
            to_concat.append(bracket)
        
        # Concatenate
        final_df = pd.concat(to_concat, ignore_index=True, sort=False)
        
        return final_df
