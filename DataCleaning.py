import pandas as pd
import glob
from datetime import datetime as dt
import pickle

# raw_data = pd.read_csv(data_root+'season 2017-2018.csv')

def parse_date(date):
    if date == '':
        return None
    else:
        return dt.strptime(date, '%d/%m/%y').date()

def parse_date2(date):
    if date == '':
        return None
    else:
        return dt.strptime(date, '%d/%m/%Y').date()

# how much goals each team get
def get_goals_scored(stat):
    teams = {}
    for i in stat.groupby('HomeTeam').mean().T.columns:
        teams[i] = []

    for i in range(len(stat)):
        HTGS = stat.iloc[i]['FTHG']
        ATGS = stat.iloc[i]['FTAG']
        teams[stat.iloc[i].HomeTeam].append(HTGS)
        teams[stat.iloc[i].AwayTeam].append(ATGS)

    GoalScored = pd.DataFrame(data=teams, index=[i for i in range(1, 39)]).T
    GoalScored[0] = 0
    for i in range(2, 39):
        GoalScored[i] = GoalScored[i] + GoalScored[i-1]

    return GoalScored

# how much goal each team conceded
def get_goals_conceded(stat):
    teams = {}
    for i in stat.groupby('HomeTeam').mean().T.columns:
        teams[i] = []

    for i in range(len(stat)):
        ATGC = stat.iloc[i]['FTHG']
        HTGC = stat.iloc[i]['FTAG']
        teams[stat.iloc[i].HomeTeam].append(HTGC)
        teams[stat.iloc[i].AwayTeam].append(ATGC)

    GoalConceded = pd.DataFrame(data=teams, index=[i for i in range(1, 39)]).T
    GoalConceded[0] = 0
    for i in range(2, 39):
        GoalConceded[i] = GoalConceded[i] + GoalConceded[i-1]
    return GoalConceded

def get_goal_info(stat):
    GS = get_goals_scored(stat)
    GC = get_goals_conceded(stat)

    j = 0
    HTGS = []
    ATGS = []
    HTGC = []
    ATGC = []

    for i in range(380):
        ht = stat.iloc[i].HomeTeam
        at = stat.iloc[i].AwayTeam
        HTGS.append(GS.loc[ht][j])
        ATGS.append(GS.loc[at][j])
        HTGC.append(GC.loc[ht][j])
        ATGC.append(GC.loc[at][j])

        # 한 주씩 확인한다고 보면 됨 (한 주에 10경기씩)
        if ((i+1)%10) == 0:
            j = j+1

    stat['HTGS'] = HTGS
    stat['ATGS'] = ATGS
    stat['HTGC'] = HTGC
    stat['ATGC'] = ATGC

    return stat

def get_match_result(stat):
    teams = {}
    for i in stat.groupby('HomeTeam').mean().T.columns:
        teams[i] = []

    for i in range(len(stat)):
        result = stat.iloc[i].FTR
        if result == 'H':
            teams[stat.iloc[i].HomeTeam].append('W')
            teams[stat.iloc[i].AwayTeam].append('L')
        elif result == 'A':
            teams[stat.iloc[i].HomeTeam].append('L')
            teams[stat.iloc[i].AwayTeam].append('W')
        else:
            teams[stat.iloc[i].HomeTeam].append('D')
            teams[stat.iloc[i].AwayTeam].append('D')

    return pd.DataFrame(data = teams, index = [i for i in range(1, 39)]).T

def change_result2point(result):
    if result == 'W':
        return 3
    elif result == 'L':
        return 0
    else:
        return 1

def get_total_points(match_result):
    match_points = match_result.applymap(change_result2point)
    for i in range(2, 39):
        match_points[i] = match_points[i] + match_points[i-1]

    match_points.insert(column = 0, loc = 0, value = [0*i for i in range(20)])
    return match_points

def get_team_points(stat):
    match_result = get_match_result(stat)
    total_pts = get_total_points(match_result)
    HTP = []
    ATP = []
    j = 0
    for i in range(380):
        ht = stat.iloc[i].HomeTeam
        at = stat.iloc[i].AwayTeam
        HTP.append(total_pts.loc[ht][j])
        ATP.append(total_pts.loc[at][j])

        if ((i+1)%10) == 0:
            j = j+1

    stat['HTP'] = HTP
    stat['ATP'] = ATP
    return stat

def get_form(stat, num):
    match_result = get_match_result(stat)
    form_final = match_result.copy()
    for i in range(num, 39):
        form_final[i] = ''
        j=0
        while j < num:
            form_final[i] += match_result[i-j]
            j+= 1
    return form_final

def add_form(stat, num):
    form = get_form(stat,num)
    h = ['M' for _ in range(num*10)]
    a = ['M' for _ in range(num*10)]

    j = num
    for i in range((num*10), 380):
        ht = stat.iloc[i].HomeTeam
        at = stat.iloc[i].AwayTeam

        past = form.loc[ht][j]
        h.append(past[num-1])

        past = form.loc[at][j]
        a.append(past[num-1])

        if ((i+1)%10) == 0:
            j = j+1

    stat['HM'+str(num)] = h
    stat['AM'+str(num)] = a

    return stat

def add_form_df(stat):
    stat_rt = add_form(stat, 1)
    stat_rt = add_form(stat_rt, 2)
    stat_rt = add_form(stat_rt, 3)
    stat_rt = add_form(stat_rt, 4)
    stat_rt = add_form(stat_rt, 5)
    return stat_rt

def get_last(stat, standing, year):
    HomeTeamLP = []
    AwayTeamLP = []
    for i in range(380):
        ht = stat.iloc[i].HomeTeam
        at = stat.iloc[i].AwayTeam
        HomeTeamLP.append(standing.loc[ht][year])
        AwayTeamLP.append(standing.loc[at][year])
    stat['HomeTeamLP'] = HomeTeamLP
    stat['AwayTeamLP'] = AwayTeamLP
    return stat

def get_matchweek(stat):
    j = 1
    MatchWeek = []
    for i in range(380):
        MatchWeek.append(j)
        if ((i+1)%10)==0:
            j = j+1
    stat['MW'] = MatchWeek
    return stat

def get_form_points(string):
    sum = 0
    for letter in string:
        sum += change_result2point(letter)
    return sum

def get_3game_ws(string):
    if string[-3:] == 'WWW':
        return 1
    else:
        return 0

def get_5game_ws(string):
    if string=='WWWWW':
        return 1
    else:
        return 0

def get_3game_ls(string):
    if string[-3:] == 'LLL':
        return 1
    else:
        return 0

def get_5game_ls(string):
    if string == 'LLLLL':
        return 1
    else:
        return 0

def only_hw(string):
    if string == 'H':
        return 'H'
    else:
        return 'NH'

# Test set : 2017-2018
# playing_stat_test
if __name__ == "__main__":
    data_root = './data/'
    file_list = []
    for file in sorted(glob.glob(data_root + 'season *.csv')):
        file_list.append(file)

    for i, file_name in enumerate(file_list[2:-1]):
        raw_data = pd.read_csv(file_name)
        if i == 7:
            raw_data.Date = raw_data.Date.apply(parse_date2)
        else:
            raw_data.Date = raw_data.Date.apply(parse_date)

        columns_req = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HTHG', 'HTAG', 'HTR', 'FTR']
        playing_statistics = raw_data[columns_req]

        playing_statistics = get_goal_info(playing_statistics)
        playing_statistics = get_team_points(playing_statistics)
        playing_statistics = add_form_df(playing_statistics)

        standings = pd.read_csv(data_root + 'EPL_standings_93-16.csv')
        standings.set_index(['Team'], inplace=True)
        standings = standings.fillna(18)

        playing_statistics = get_last(playing_statistics, standings,i+2)
        playing_statistics = get_matchweek(playing_statistics)

        playing_statistics['HTFormPtsStr'] = playing_statistics['HM1'] + playing_statistics['HM2'] + playing_statistics[
            'HM3'] + playing_statistics['HM4'] + playing_statistics['HM5']
        playing_statistics['ATFormPtsStr'] = playing_statistics['AM1'] + playing_statistics['AM2'] + playing_statistics[
            'AM3'] + playing_statistics['AM4'] + playing_statistics['AM5']

        playing_statistics['HTFormPts'] = playing_statistics['HTFormPtsStr'].apply(get_form_points)
        playing_statistics['ATFormPts'] = playing_statistics['ATFormPtsStr'].apply(get_form_points)

        playing_statistics['HTWinStreak3'] = playing_statistics['HTFormPtsStr'].apply(get_3game_ws)
        playing_statistics['HTWinStreak5'] = playing_statistics['HTFormPtsStr'].apply(get_5game_ws)
        playing_statistics['HTLossStreak3'] = playing_statistics['HTFormPtsStr'].apply(get_3game_ls)
        playing_statistics['HTLossStreak5'] = playing_statistics['HTFormPtsStr'].apply(get_5game_ls)

        playing_statistics['ATWinStreak3'] = playing_statistics['ATFormPtsStr'].apply(get_3game_ws)
        playing_statistics['ATWinStreak5'] = playing_statistics['ATFormPtsStr'].apply(get_5game_ws)
        playing_statistics['ATLossStreak3'] = playing_statistics['ATFormPtsStr'].apply(get_3game_ls)
        playing_statistics['ATLossStreak5'] = playing_statistics['ATFormPtsStr'].apply(get_5game_ls)

        # Get Goal Difference
        playing_statistics['HTGD'] = playing_statistics['HTGS'] - playing_statistics['HTGC']
        playing_statistics['ATGD'] = playing_statistics['ATGS'] - playing_statistics['ATGC']

        # Difference in Points
        playing_statistics['DiffPts'] = playing_statistics['HTP'] - playing_statistics['ATP']
        playing_statistics['DiffFormPts'] = playing_statistics['HTFormPts'] - playing_statistics['ATFormPts']

        # Difference in last year position
        playing_statistics['DiffLP'] = playing_statistics['HomeTeamLP'] - playing_statistics['AwayTeamLP']

        # Scale DiffPts, DiffFormPts, HTGD, ATGD by match week
        cols = ['HTGD', 'ATGD', 'DiffPts', 'DiffFormPts', 'HTP', 'ATP']
        playing_statistics.MW = playing_statistics.MW.astype(float)

        for col in cols:
            playing_statistics[col] = playing_statistics[col] / playing_statistics.MW

        playing_statistics['FTR'] = playing_statistics.FTR.apply(only_hw)
        with open('stat_{}.pkl'.format(i+1995), 'wb') as handle:
            pickle.dump(playing_statistics, handle, protocol=pickle.HIGHEST_PROTOCOL)
