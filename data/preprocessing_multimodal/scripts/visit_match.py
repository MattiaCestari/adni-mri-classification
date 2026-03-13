import pandas as pd

from tqdm import tqdm

def date_match(left, right, id_col_left, date_col_left, date_col_right, condition, match,  tolerance, interpolate=False): 
    """
    left: left DataFrame.
    right: right DataFrame.
    id_col_left: column of left dataframe to be used as unique id.
    date_col_left: date column on left dataframe.
    date_col_right: date column on right dataframe.
    match: column from right dataframe to be matched.
    condition: column whose value must match between left and right dataframe (e.g. subject).
    tolerance: dates matching tolerance in days.
    interpolate: if closest visit is not under tolerance, try checking for earlier and later visit with same value. 

    """

    if not isinstance(match, list):
        match = [match]

    # Make a copy of the dataframes
    left = left.copy()
    right = right.copy()

    # Cast date columns to datetime 
    left[date_col_left] = pd.to_datetime(left[date_col_left])
    right[date_col_right] = pd.to_datetime(right[date_col_right])

    # dfs to concatenate later for 
    concat = []

    # Loop on condition 
    for cond in tqdm(left[ condition ].unique()): 

        # Filter for cond on both dfs
        left_cond = left[ left[ condition ] == cond].copy()
        right_cond = right[ right[ condition ] == cond].copy()

        # Drop NaN values in date columns 
        left_cond = left_cond.dropna(subset=[date_col_left])
        right_cond = right_cond.dropna(subset=[date_col_right])

        # Sort both dfs by dates 
        left_cond = left_cond.sort_values(date_col_left)
        right_cond = right_cond.sort_values(date_col_right)

        # Match left rows with temporally closest right rows within tolerance
        nearest = pd.merge_asof(
            left_cond,
            right_cond[ [date_col_right] + match ],
            left_on=date_col_left,
            right_on=date_col_right,
            direction="nearest",
            tolerance=pd.Timedelta(days=tolerance),
        )

        # Filter not NaN
        matched = nearest[ nearest[match].notna().all(axis=1)].copy()

        # if every row on the left df has been matched
        if (len(matched) == len(left_cond)) or (not interpolate): 
            concat.append(matched)
            continue # Continue to next conditioner 

        # Otherwise attempt interpolation: 

        # Remaining left rows (not matched)
        left_cond_remaining = left_cond[ ~left_cond[id_col_left].isin(set(matched[id_col_left])) ].copy()

        # Get closest match before 
        before = pd.merge_asof(
            left_cond_remaining,
            right_cond[[date_col_right] + match],
            left_on=date_col_left,
            right_on=date_col_right,
            direction="backward", 
            allow_exact_matches=True,
        )

        # Get closest visits after
        after = pd.merge_asof(
            left_cond_remaining,
            right_cond[[date_col_right] + match],
            left_on=date_col_left,
            right_on=date_col_right,
            direction="forward",    
            allow_exact_matches=True,
        )

        # Rename date col and  match cols after
        match_after = [ m+"_after" for m in match]
        date_col_right_after = date_col_right + "_after"

        after = after.rename(columns={date_col_right: date_col_right_after})
        after = after.rename(columns={ match[i]:match_after[i] for i in range(len(match))}) 

        # Bracket: closest visit before + closest visit after!
        bracket = before.merge(
            after[match_after + [id_col_left, date_col_right_after]],
            on=id_col_left,
            how="inner",
        )

        # both sides must be present
        valid_before = bracket[match].notna().all(axis=1)
        valid_after = bracket[match_after].notna().all(axis=1)

        # all match columns must agree pairwise
        same = pd.Series(True, index=bracket.index)
        for m in match:
            same &= bracket[m] == bracket[f"{m}_after"]

        bracket = bracket[valid_before & valid_after & same].copy()

        # Keep closest exam date
        d_before = (bracket[date_col_left] - bracket[date_col_right]).abs()
        d_after  = (bracket[date_col_left] - bracket[date_col_right_after]).abs()
        bracket[date_col_right] = bracket[date_col_right].where(d_before <= d_after, bracket[date_col_right_after])

        # Drop unused columns 
        bracket = bracket.drop(columns=match_after+[date_col_right_after])

        concat.append(bracket)

    return pd.concat(concat, ignore_index=True, sort=False)