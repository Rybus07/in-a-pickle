import numpy as np
import pandas as pd
import argparse

def remove_incomplete_rally(df):
    # finding and removing incomplete rallies
    # identifying incomplete rallies if rally_len does not match the number of shots
    inds = df.index
    prev_rally_id = df.loc[inds[0], 'rally_id']
    incomplete_rallies = []
    cur_shot_count = 0

    # loop through df
    for i, ind in enumerate(inds):
        cur_rally_id = df.loc[ind, 'rally_id']
        # if enter new rally, check if shot_count matches rally_len of prev rally
        if cur_rally_id != prev_rally_id:
            # if shot_count does not match, then add to list of incomplete rallies
            if cur_shot_count != df.loc[inds[i-1], 'rally_len']:
                incomplete_rallies.append(prev_rally_id)
            # either case, start tracking for new rally_id
            prev_rally_id = cur_rally_id
            cur_shot_count = 1
        # if in same rally, increment cur_shot_count
        else:
            cur_shot_count += 1

    # now removing incomplete rallies
    incomplete_rally_mask = []
    for i in df.index:
        if df.loc[i, 'rally_id'] in incomplete_rallies:
            incomplete_rally_mask.append(False)
        else:
            incomplete_rally_mask.append(True)

    return df[incomplete_rally_mask]

def add_shot_outcome(df):
    '''
    Creates column shot_outcome by looking at surrounding shot_types
    shot_outcome can be winner, extend rally, force error, create opportunity,
        pop up, or unforced error
    NOTE: ASSUMES DATAFRAME IS IN DESCENDING ORDER OF SHOT NUMBER.
    ALSO, THIS FUNCTION MAY NEED SOME WORK
    '''
    shot_outcome = []

    inds = df.index
    for i in range(len(inds)):
        if df.loc[inds[i], 'shot_nbr'] == df.loc[inds[i], 'rally_len']:
            shot_outcome.append(df.loc[inds[i], 'ending_type'])
        elif shot_outcome[-1] == 'Pop Up':
            shot_outcome.append('Create Opportunity')
        else:
            if shot_outcome[-1] == 'Error':
                shot_outcome.append('Force Error')
            elif df.loc[inds[i-1], 'shot_type'] == 'HB' and df.loc[inds[i], 'shot_type'] == 'D':
                shot_outcome.append('Pop Up')
            elif df.loc[inds[i], 'shot_type'] == 'SE':
                shot_outcome.append('Start Rally')
            elif df.loc[inds[i], 'shot_type'] == 'tsDrv' and shot_outcome[-1] in ['Unforced Error', 'Error']:
                shot_outcome.append('Winner')
            elif df.loc[inds[i], 'shot_type'] == 'SP':
                cond1 = df.loc[inds[i], 'shot_nbr'] + 2 == df.loc[inds[i], 'rally_len']
                cond2 = df.loc[inds[i-2], 'ending_type'] == 'Winner'
                if cond1 and cond2:
                    shot_outcome.append('Create Opportunity')
                elif df.loc[inds[i-1], 'ending_type'] == 'Winner':
                    shot_outcome.append('Lose Rally')
                elif df.loc[inds[i-1], 'shot_type'] == 'HB':
                    shot_outcome.append('Initiate Hands Battle')
                else:
                    shot_outcome.append('Speed Up')
            else:
                shot_outcome.append('Extend Rally')

    df['shot_outcome'] = shot_outcome
    return df

def add_rally_outcome(df):
    '''
    Creates column rally_outcome by looking at who won the rally and surrounding rally
    rally_outcome can be point, next server, sideout, or game over
    NOTE: ASSUMES DATAFRAME IS IN DESCENDING ORDER
    '''
    # initializing 0
    # since descending order, first row is always game over
    rally_outcome = ['Game Over']

    inds = df.index
    prev_rally_id = df.iloc[0]['rally_id']
    prev_game_id = df.iloc[0]['game_id']
    for i in range(1, len(inds)):
        cur_rally_id = df.loc[inds[i], 'rally_id']
        cur_game_id = df.loc[inds[i], 'game_id']
        # if new game, then push game over since descending order
        if cur_game_id != prev_game_id:
            rally_outcome.append('Game Over')
            # update prev_game_id and prev_rally_id
            prev_game_id = cur_game_id
            prev_rally_id = cur_rally_id
        # if in the same rally, then continue pushing the outcome until we hit a new rally
        elif cur_rally_id == prev_rally_id:
            rally_outcome.append(rally_outcome[-1])
        else:
            # same game, at the end of a new rally since descending
            if df.loc[inds[i], 'srv_team_id'] == df.loc[inds[i], 'w_team_id']:
                rally_outcome.append('Point')
            # if serving team did not win rally, can either be next server or side out
            else:
                # if new serving team relative to last rally, then side out
                if df.loc[inds[i-1], 'srv_team_id'] != df.loc[inds[i], 'srv_team_id']:
                    rally_outcome.append('Side Out')
                else:
                    rally_outcome.append('Next Server')
            prev_rally_id = cur_rally_id

    df['rally_outcome'] = rally_outcome
    
    return df

def change_coordinate_system(df):
    '''
    Purpose is to change coordinate system of loc_x, loc_y, next_loc_x, next_loc_y
    such that the net is at (0, 0) to (20, 0), the serving team starts at y=-22,
    and the returning team starts at y=22. See plotting function for visualization
    of rallies
    '''
    inds = df.index
    for i in inds:
        if df.loc[i, 'shot_nbr'] % 2 == 1:
            df.loc[i, 'loc_x'] = 20 - df.loc[i, 'loc_x']
            df.loc[i, 'loc_y'] = -df.loc[i, 'loc_y']
        else:
            df.loc[i, 'next_loc_x'] = 20 - df.loc[i, 'next_loc_x']
            df.loc[i, 'next_loc_y'] = -df.loc[i, 'next_loc_y']

    return df

def standardize_court_loc(df):
  '''
  The purpose of this function is to standardize court locations to be in a 1 by 1.1 court instead of a 20 by 44 dimension,
  reducting the effects of what 
  '''
  df = df.copy()
  df['loc_x'] = df['loc_x'] / 20
  df['loc_y'] = df['loc_y'] / 20
  df['next_loc_x'] = df['next_loc_x'] / 20
  df['next_loc_y'] = df['next_loc_y'] / 20
  return df

def delta_x_loc(df):
        """
        This function is used to calculate the change in x drirection for each shot
        """
        df = df.copy()
        df['delta_x_loc'] = df['next_loc_x'] - df['loc_x']
        return df
    
def delta_y_loc(df):
    """
    This function is used to calculate the change in y drirection for each shot
    """
    df = df.copy()
    df['delta_y_loc'] = df['next_loc_y'] - df['loc_y']
    return df

def add_shot_angle(df):
    """
    Calculates the angle of the shot in radians using arctan2.
    Handles the trajectory from current location to next location.
    """
    df = df.copy()

    # 1. Calculate the coordinate deltas
    dy = df['next_loc_y'] - df['loc_y']
    dx = df['next_loc_x'] - df['loc_x']

    # 2. Use arctan2 (y, x) for quadrant-aware angles (-pi to pi)
    # This correctly identifies shots moving 'backward' or 'sideways'
    df['shot_angle'] = np.arctan2(dy, dx)

    return df

def add_kitchen_distance(df):
    '''
    This function calculates how far from the kitchen line a ball has been played from, the aim for this function
    is to help further separate shots played from the kitchen line or not 
    '''
    df = df.copy()

    # 1. Identify which side of the net the ball is on (7 or -7)
    # 2. Subtract the ball's y-position from that line
    # 3. Use abs() to ensure the distance is always positive
    df['dist_to_kitchen'] = df['loc_y'].apply(lambda y: abs(y - 7) if y > 0 else abs(y + (-7)))

    # Optional: If the ball is INSIDE the kitchen, set distance to 0
    # (i.e., if |y| < 7, the ball is in the kitchen)
    df.loc[df['loc_y'].abs() < 7, 'dist_to_kitchen'] = 0

    return df

'''
Steps for cleaning and transforming shot data in main():
    1. Load shot and rally data
    2. Merge with rally data. Then sort and reset index
       Columns to merge = [rally_id, game_id, match_id, rally_nbr,
       rally_len, w_team_id, srv_team_id, rtrn_team_id, ending_type]
    3. Remove edge cases
        a. Remove any rallies with a missed serve.
           Identified where rally_len == 1 and loc_y is NaN
    4. Fill NaNs
        a. Fill NaNs in loc_y
            i. Backfill NaN in loc_y using next_loc_y from previous shot
               Note: Since step 2 removes rallies with a missing serve
               (loc_y == NaN and rally_len == 1), all NaNs in loc_y
               will have a previous shot. This will also handle all NaNs in loc_y
        b. Fill NaNs in next_loc_x, next_loc_y using next shot if present
           Note: remaining NaNs can now be assumed to be at the end of a rally
        c. Fill NaNs in shot_type with U for unknown
    5. Remove incomplete rallies. Identified where max(shot_nbr) != rally_len
    6. Adding additional columns
        a. add shot_outcome MAKE SURE DATAFRAME IS IN DESCENDING ORDER OF SHOT NBR
        b. add rally_outcome MAKE SURE DATAFRAME IS IN DESCENDING ORDER OF SHOT NBR
        c. add delta_x_loc, delta_y_loc for change in x, y coordinates
        d. add shot_angle
        e. add dist_to_kitchen to represent the distance a shot was played wrt kitchen
    7. Fill additional NaNs
        a. Fill NaNs in ending_type, shot_outcome, and shot_type
    8. Change coordinate system
    9. Standardize the court coordinate system
    10. Saving to output directory
'''
def main(args):
    # Step 1: Load shot and rally data
    shot_df = pd.read_csv(args.i_dir+args.shot_file)
    rally_df = pd.read_csv(args.i_dir+args.rally_file)

    # Step 2: Merge with rally data. Then sort and reset index
    # Note: order will be descending in shot_nbr
    rally_cols = ['rally_id', 'game_id', 'match_id', 'rally_nbr', 'rally_len',
                  'w_team_id', 'srv_team_id', 'rtrn_team_id', 'ending_type']
    shot_rally_df = pd.merge(shot_df, rally_df[rally_cols], on='rally_id', how='left')

    new_order = ['match_id', 'game_id', 'rally_id', 'rally_nbr', 'rally_len',
                 'srv_team_id', 'rtrn_team_id', 'w_team_id', 'shot_id',
                 'shot_nbr', 'shot_type', 'player_id', 'loc_x', 'loc_y',
                 'next_loc_x', 'next_loc_y', 'ending_type']
    shot_rally_df = shot_rally_df[new_order]
    shot_rally_df.sort_values(['match_id', 'game_id', 'rally_nbr', 'shot_nbr'],
                              inplace=True, ascending=False)
    shot_rally_df = shot_rally_df.astype({'rally_len':'int64'})
    shot_rally_df.reset_indexx(drop=True, inplace=True)

    # Step 3: Fill edge case where loc_y == NaN and rally_len == 1
    # Case for missed serve, filling loc_y with 23.0, value will be flipped later
    cond1 = [x==1 for x in shot_rally_df['rally_len']]
    cond2 = shot_rally_df['loc_y'].isnull()
    mask = [(a and b) for a,b in zip(cond1, cond2)]
    shot_rally_df.loc[mask, 'loc_y'] = 23.0

    # Step 4: Filling NaNs
    # 4a. Backfilling NaNs in loc_y
    inds = shot_rally_df.index
    for i in inds:
        if shot_rally_df.loc[i, 'loc_y'].isnull():
            shot_rally_df.loc[i, 'loc_y'] = shot_df.loc[i+1, 'next_loc_y']

    # 4b. Fill NaNs in next_loc_x, next_loc_y, using next shot loc_x, loc_y
    cond1 = shot_rally_df['next_loc_x'].isnull()
    cond2 = shot_rally_df['next_loc_y'].isnull()
    mask = [(a and b) for a, b in zip(cond1, cond2)]

    # for each pair of nans, check to see if there is a valid subsequent shot
    # if there is, then pull values from there
    inds = shot_rally_df[mask].index

    for i in inds:
        if i+1 < len(shot_rally_df):
            if (shot_rally_df.loc[i+1, 'shot_nbr'] == shot_rally_df.loc[i, 'shot_nbr'] + 1):
                shot_rally_df.loc[i, 'next_loc_x'] = shot_rally_df.loc[i+1, 'loc_x']
                shot_rally_df.loc[i, 'next_loc_y'] = shot_rally_df.loc[i+1, 'loc_y']

    # Step 5: Removing rallies where where max(shot_nbr) != rally_len
    # See function above
    shot_rally_df = remove_incomplete_rally(shot_rally_df)

    # Step 6: Adding additional columns
    # 6a. add shot_outcome
    shot_rally_df = add_shot_outcome(shot_rally_df)
    # 6b. add rally_outcome
    shot_rally_df = add_rally_outcome(shot_rally_df)
    # 6c. add delta_x_loc, delta_y_loc
    shot_rally_df = delta_x_loc(shot_rally_df)
    shot_rally_df = delta_y_loc(shot_rally_df)
    # 6d. add shot_angle
    shot_rally_df = add_shot_angle(shot_rally_df)
    # 6e. add dist_to_kitchen
    shot_rally_df = add_kitchen_distance(shot_rally_df)

    # Step 7: Fill NaNs in ending_type, shot_outcome, and shot_type with unknown
    shot_rally_df.fillna(value = {'ending_type':'Unknown',
                                  'shot_outcome':'Unknown',
                                  'shot_type':'Unknown'},
                         inplace=True)

    # Step 8: Change coordinate system
    # serving team will start at y < -22
    # returning team will start at y > 22
    shot_rally_df = change_coordinate_system(shot_rally_df)

    # Step 9: Standardizing the court coordinate system
    shot_rally_df = standardize_court_loc(shot_rally_df)

    # Step 10: Saving to output directory
    shot_rally_df.to_csv(args.o_dir+'/'+args.save_file, index=True)
    
    return

if __name__ == '__main__':
    # writing command-line interface
    desc = """Script for cleaning and transforming shot.csv.
          Also pulls information from rally.csv"""

    parser = argparse.ArgumentParser(prog='cleaning', 
                                     description=desc)

    parser.add_argument('-i', '--i_dir', help='directory containing input file', default='./')
    parser.add_argument('-o', '--o_dir', help='directory for storing output', default='./')
    parser.add_argument('-s', '--shot_file', help='name of file holding shot data, should be csv',
                        default='shot.csv')
    parser.add_argument('-r', '--rally_file', help='name of file holding rally data, should be csv',
                        default='rally.csv')
    parser.add_argument('-s', '--save_file', help='name of file to save cleaned data to',
                        default='shot_rally.csv')
    
    args = parser.parse_args()
    
    main(args)
