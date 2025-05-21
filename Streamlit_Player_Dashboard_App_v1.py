from functools import lru_cache
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import statistics
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")
import requests
from bs4 import BeautifulSoup, Comment
import os
from pathlib import Path
import time
from scipy import stats
from statistics import mean
from math import pi
import streamlit as st
from sklearn.preprocessing import StandardScaler
from urllib.request import urlopen
import matplotlib.pyplot as plt
from PIL import Image
from mplsoccer import PyPizza, add_image, FontManager
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.cm as cm
import matplotlib
import gdown

root = os.getcwd() + '/'

url = 'https://www.dropbox.com/scl/fi/nsqh1o4lywfs5it9buu88/Final-FBRef-All-Leagues.csv?rlkey=jhci52hjq0lw8xjnyhydng3uj&st=ujr89d43&raw=1'

df = pd.read_csv(url)

#st.write(df.head())

#https://drive.google.com/uc?id=1AZhc9HS0IBH0FMC7NArwRP3eu6n14Bv2&export=download

#df = pd.read_csv(download_url)

#read csv file
#df = pd.read_csv("/Users/scini/Documents - Local/Final FBRef All Leagues.csv")

#drop unneeded fields and rename position field imported from external csv file
df = df.drop(['UrlFBref', 'UrlTmarkt'], axis=1)
df = df.rename(columns={'TmPos': 'Main Position'})

def map_position_group(position):
    if position in ['Left-Back', 'Right-Back']:
        return 'FB'
    elif position == 'Centre-Back':
        return 'CB'
    elif position == 'Defensive Midfield':
        return 'DM'
    elif position == 'Central Midfield':
        return 'CM'
    elif position in ['Attacking Midfield', 'Second Striker']:
        return 'AM'
    elif position in ['Left Winger', 'Left Midfield', 'Right Winger', 'Right Midfield']:
        return 'W'
    elif position == 'Centre-Forward':
        return 'ST'

def map_null_main_position(position):
    if position == 'FB':
        return 'Full-Back'
    elif position == 'CB':
        return 'Centre-Back'
    elif position == 'DM':
        return 'Defensive Midfield'
    elif position == 'CM':
        return 'Central Midfield'
    elif position == 'AM':
        return 'Attacking Midfield'
    elif position == 'W':
        return 'Winger'
    elif position == 'ST':
        return 'Centre-Forward'

def map_null_position_group(position):
    if position == 'Full-Back':
        return 'FB'
    elif position == 'Centre-Back':
        return 'CB'
    elif position == 'Defensive Midfield':
        return 'DM'
    elif position == 'Central Midfield':
        return 'CM'
    elif position == 'Attacking Midfield':
        return 'AM'
    elif position == 'Winger':
        return 'W'
    elif position == 'Centre-Forward':
        return 'ST'

def map_null_position_group_fb(position):
    if position == 'FB':
        return 'FB'

def map_null_position_group_cb(position):
    if position == 'CB':
        return 'CB'

def map_null_position_group_dm(position):
    if position == 'DM':
        return 'DM'

def map_null_position_group_cm(position):
    if position == 'CM':
        return 'CM'

def map_null_position_group_am(position):
    if position == 'AM':
        return 'AM'

def map_null_position_group_w(position):
    if position == 'W':
        return 'W'

def map_null_position_group_st(position):
    if position == 'ST':
        return 'ST'

# Apply the mapping to create new column
df['Position Group'] = df['Main Position'].apply(map_position_group)

#Apply new player names for easier selection
df['Player Name'] = df['Player'] + ' (' + df['Squad'] + ')'

df['Extract'] = df['Extract'].fillna('Next 12 Leagues')

df['OffPer90'] = df['Off']/(df['Min'] / 90)
df['PKwonPer90'] = df['PKwon']/(df['Min'] / 90)
df['PKconPer90'] = df['PKcon']/(df['Min'] / 90)
df['OGPer90'] = df['OG']/(df['Min'] / 90)
df['RecovPer90'] = df['Recov']/(df['Min'] / 90)
df['AerialWinsPer90'] = df['AerialWins']/(df['Min'] / 90)
df['AerialLossPer90'] = df['AerialLoss']/(df['Min'] / 90)

df['LineBreakingPassesPer90'] = (df['ProgPassesPer90'] - (df['ThruBallsPer90'] + df['LongPassCmpPer90'] + (df['PenAreaCmpPer90'] * 0.3)))
df['LineBreakingPassesPer90'] = np.where(df['LineBreakingPassesPer90'] == 0, 0.1, df['LineBreakingPassesPer90'])
df['LineBreakingPass%'] = (df['LineBreakingPassesPer90'] / df['PassesAttemptedPer90']) * 100

df['LooseBallWinsPer90'] = (df['RecovPer90'] - (df['TklWinPossPer90'] + df['IntPer90'] + (df['PassBlocksPer90'] * 0.5)))
df['LooseBallWinsPer90'] = np.where(df['LooseBallWinsPer90'] == 0, 0.1, df['LooseBallWinsPer90'])

df_combined = df.copy()

def create_percentile_rankings(position, additional_player, df):

    df = df_combined.copy()

    df.loc[
        (df['Player Name'] == additional_player) & (df['Main Position'].isnull()),
        'Main Position'
    ] = map_null_main_position(position)

    #If player's Position Group is still null, recalculate it from updated Main Position
    df.loc[
        (df['Player Name'] == additional_player) & (df['Position Group'].isnull()),
        'Position Group'
    ] = df.loc[
        (df['Player Name'] == additional_player) & (df['Position Group'].isnull()),
        'Main Position'
    ].apply(map_null_position_group)

    df = df[(df['Position Group'] == position) | (df['Player Name'] == additional_player)]

    metrics_to_rank = [
    'Min', 'G+A', 'Glsxx', 'Goals', 'Shots', 'SoT', 'SoT%', 'Sh/90', 'SoT/90', 'G/Sh', 'G/SoT', 'AvgShotDistance', 'FKShots', 'PK', 'PKsAtt', 'xG', 'npxG', 'npxG/Sh', 'G-xG', 'npG-xG', 'PassesCompleted', 'PassesAttempted', 'TotCmp%', 'TotalPassDist', 'ProgPassDist', 'ShortPassCmp', 'ShortPassAtt', 'ShortPassCmp%', 'MedPassCmp', 'MedPassAtt', 'MedPassCmp%', 'LongPassCmp', 'LongPassAtt', 'LongPassCmp%', 'Assists', 'xAG', 'xA', 'A-xAG', 'KeyPasses', 'Final1/3Cmp', 'PenAreaCmp', 'CrsPenAreaCmp', 'ProgPasses', 'LivePass', 'DeadPass', 'FKPasses', 'ThruBalls', 'Switches', 'Crs', 'ThrowIn', 'CK', 'InSwingCK', 'OutSwingCK', 'StrCK', 'Cmpxxx', 'PassesToOff', 'PassesBlocked', 'SCA', 'SCA90', 'SCAPassLive', 'SCAPassDead', 'SCADrib', 'SCASh', 'SCAFld', 'SCADef', 'GCA', 'GCA90', 'GCAPassLive', 'GCAPassDead', 'GCADrib', 'GCASh', 'GCAFld', 'GCADef', 'Tkl', 'TklWinPoss', 'Def3rdTkl', 'Mid3rdTkl', 'Att3rdTkl', 'DrbTkl', 'DrbPastAtt', 'DrbTkl%', 'DrbPast', 'Blocks', 'ShBlocks', 'PassBlocks', 'Int', 'Tkl+Int', 'Clr', 'Err', 'Touches', 'DefPenTouch', 'Def3rdTouch', 'Mid3rdTouch', 'Att3rdTouch', 'AttPenTouch', 'LiveTouch', 'AttDrb', 'SuccDrb', 'DrbSucc%', 'TimesTackled', 'TimesTackled%', 'Carries', 'TotalCarryDistance', 'ProgCarryDistance', 'ProgCarries', 'CarriesToFinalThird', 'CarriesToPenArea', 'CarryMistakes', 'Disposesed', 'ReceivedPass', 'ProgPassesRec', 'Yellows', 'Reds', 'Yellow2', 'Fls', 'Fld', 'Off', 'PKwon', 'PKcon', 'OG', 'Recov', 'AerialWins', 'AerialLoss', 'AerialWin%', 'G+APer90', 'GlsxxPer90', 'GoalsPer90', 'ShotsPer90', 'SoTPer90', 'SoT%Per90', 'Sh/90Per90', 'SoT/90Per90', 'G/ShPer90', 'G/SoTPer90', 'AvgShotDistancePer90', 'FKShotsPer90', 'PKPer90', 'PKsAttPer90', 'xGPer90', 'npxGPer90', 'npxG/ShPer90', 'G-xGPer90', 'npG-xGPer90', 'PassesCompletedPer90', 'PassesAttemptedPer90', 'TotCmp%Per90', 'TotalPassDistPer90', 'ProgPassDistPer90', 'ShortPassCmpPer90', 'ShortPassAttPer90', 'ShortPassCmp%Per90', 'MedPassCmpPer90', 'MedPassAttPer90', 'MedPassCmp%Per90', 'LongPassCmpPer90', 'LongPassAttPer90', 'LongPassCmp%Per90', 'AssistsPer90', 'xAGPer90', 'xAPer90', 'A-xAGPer90', 'KeyPassesPer90', 'Final1/3CmpPer90', 'PenAreaCmpPer90', 'CrsPenAreaCmpPer90', 'ProgPassesPer90', 'LivePassPer90', 'DeadPassPer90', 'FKPassesPer90', 'ThruBallsPer90', 'SwitchesPer90', 'CrsPer90', 'ThrowInPer90', 'CKPer90', 'InSwingCKPer90', 'OutSwingCKPer90', 'StrCKPer90', 'CmpxxxPer90', 'PassesToOffPer90', 'PassesBlockedPer90', 'SCAPer90', 'SCA90Per90', 'SCAPassLivePer90', 'SCAPassDeadPer90', 'SCADribPer90', 'SCAShPer90', 'SCAFldPer90', 'SCADefPer90', 'GCAPer90', 'GCA90Per90', 'GCAPassLivePer90', 'GCAPassDeadPer90', 'GCADribPer90', 'GCAShPer90', 'GCAFldPer90', 'GCADefPer90', 'TklPer90', 'TklWinPossPer90', 'Def3rdTklPer90', 'Mid3rdTklPer90', 'Att3rdTklPer90', 'DrbTklPer90', 'DrbPastAttPer90', 'DrbTkl%Per90', 'DrbPastPer90', 'BlocksPer90', 'ShBlocksPer90', 'PassBlocksPer90', 'IntPer90', 'Tkl+IntPer90', 'ClrPer90', 'ErrPer90', 'TouchesPer90', 'DefPenTouchPer90', 'Def3rdTouchPer90', 'Mid3rdTouchPer90', 'Att3rdTouchPer90', 'AttPenTouchPer90', 'LiveTouchPer90', 'AttDrbPer90', 'SuccDrbPer90', 'DrbSucc%Per90', 'TimesTackledPer90', 'TimesTackled%Per90', 'CarriesPer90', 'TotalCarryDistancePer90', 'ProgCarryDistancePer90', 'ProgCarriesPer90', 'CarriesToFinalThirdPer90', 'CarriesToPenAreaPer90', 'CarryMistakesPer90', 'DisposesedPer90', 'ReceivedPassPer90', 'ProgPassesRecPer90', 'YellowsPer90', 'RedsPer90', 'Yellow2Per90', 'FlsPer90', 'FldPer90', 'OffPer90', 'PKwonPer90', 'PKconPer90', 'OGPer90', 'RecovPer90', 'AerialWinsPer90', 'AerialLossPer90', 'AerialWin%Per90', '90sPer90', 'AvgTeamPoss', 'OppTouches', 'TeamMins', 'TeamTouches90', 'pAdjTkl+IntPer90', 'pAdjClrPer90', 'pAdjShBlocksPer90', 'pAdjPassBlocksPer90', 'pAdjIntPer90', 'pAdjDrbTklPer90', 'pAdjTklWinPossPer90', 'pAdjDrbPastPer90', 'pAdjAerialWinsPer90', 'pAdjAerialLossPer90', 'pAdjDrbPastAttPer90', 'TouchCentrality', 'Tkl+IntPer600OppTouch', 'pAdjTouchesPer90', 'CarriesPer50Touches', 'ProgCarriesPer50Touches', 'ProgPassesPer50CmpPasses', 'ProgDistancePerCarry', 'ProgCarryEfficiency', 'PlayerFBref', 'ShortPass%', 'MediumPass%', 'LongPass%', 'ProgPass%', 'Switch%', 'KeyPass%', 'Final3rdPass%', 'ThroughPass%', 'Def3rdTouch%', 'Mid3rdTouch%', 'Att3rdTouch%', 'AttPenTouch%', 'ActionsPerTouch', 'Def3rdTkl%', 'Mid3rdTkl%', 'Att3rdTkl%', 'LineBreakingPassesPer90', 'LineBreakingPass%', 'LooseBallWinsPer90'
    ]

    for metric in metrics_to_rank:
        # Create the percentile column name
        percentile_col = f'{metric}_PR'
        
        # Calculate percentile rank
        df[percentile_col] = df.groupby(['Season', 'Extract'])[metric].rank(pct=True) * 100
        
        # Round to 1 decimal place
        df[percentile_col] = df[percentile_col].round(1)

    # Drop all duplicates
    df = df.sort_values('Min', ascending=False).drop_duplicates(subset=['Player Name', 'Squad', 'Season', 'Main Position'], keep='first')

    return df

def create_percentile_rankings_comparison(position, additional_player, additional_player2, df):

    df = df_combined.copy()

    df.loc[
        (df['Player Name'] == additional_player) & (df['Main Position'].isnull()),
        'Main Position'
    ] = map_null_main_position(position)

    df.loc[
        (df['Player Name'] == additional_player) & (df['Position Group'].isnull()),
        'Position Group'
    ] = df.loc[
        (df['Player Name'] == additional_player) & (df['Position Group'].isnull()),
        'Main Position'
    ].apply(map_null_position_group)

    df.loc[
        (df['Player Name'] == additional_player2) & (df['Main Position'].isnull()),
        'Main Position'
    ] = map_null_main_position(position)

    df.loc[
        (df['Player Name'] == additional_player2) & (df['Position Group'].isnull()),
        'Position Group'
    ] = df.loc[
        (df['Player Name'] == additional_player2) & (df['Position Group'].isnull()),
        'Main Position'
    ].apply(map_null_position_group)

    df = df[(df['Position Group'] == position) | (df['Player Name'] == additional_player) | (df['Player Name'] == additional_player2)]

    metrics_to_rank = [
    'Min', 'G+A', 'Glsxx', 'Goals', 'Shots', 'SoT', 'SoT%', 'Sh/90', 'SoT/90', 'G/Sh', 'G/SoT', 'AvgShotDistance', 'FKShots', 'PK', 'PKsAtt', 'xG', 'npxG', 'npxG/Sh', 'G-xG', 'npG-xG', 'PassesCompleted', 'PassesAttempted', 'TotCmp%', 'TotalPassDist', 'ProgPassDist', 'ShortPassCmp', 'ShortPassAtt', 'ShortPassCmp%', 'MedPassCmp', 'MedPassAtt', 'MedPassCmp%', 'LongPassCmp', 'LongPassAtt', 'LongPassCmp%', 'Assists', 'xAG', 'xA', 'A-xAG', 'KeyPasses', 'Final1/3Cmp', 'PenAreaCmp', 'CrsPenAreaCmp', 'ProgPasses', 'LivePass', 'DeadPass', 'FKPasses', 'ThruBalls', 'Switches', 'Crs', 'ThrowIn', 'CK', 'InSwingCK', 'OutSwingCK', 'StrCK', 'Cmpxxx', 'PassesToOff', 'PassesBlocked', 'SCA', 'SCA90', 'SCAPassLive', 'SCAPassDead', 'SCADrib', 'SCASh', 'SCAFld', 'SCADef', 'GCA', 'GCA90', 'GCAPassLive', 'GCAPassDead', 'GCADrib', 'GCASh', 'GCAFld', 'GCADef', 'Tkl', 'TklWinPoss', 'Def3rdTkl', 'Mid3rdTkl', 'Att3rdTkl', 'DrbTkl', 'DrbPastAtt', 'DrbTkl%', 'DrbPast', 'Blocks', 'ShBlocks', 'PassBlocks', 'Int', 'Tkl+Int', 'Clr', 'Err', 'Touches', 'DefPenTouch', 'Def3rdTouch', 'Mid3rdTouch', 'Att3rdTouch', 'AttPenTouch', 'LiveTouch', 'AttDrb', 'SuccDrb', 'DrbSucc%', 'TimesTackled', 'TimesTackled%', 'Carries', 'TotalCarryDistance', 'ProgCarryDistance', 'ProgCarries', 'CarriesToFinalThird', 'CarriesToPenArea', 'CarryMistakes', 'Disposesed', 'ReceivedPass', 'ProgPassesRec', 'Yellows', 'Reds', 'Yellow2', 'Fls', 'Fld', 'Off', 'PKwon', 'PKcon', 'OG', 'Recov', 'AerialWins', 'AerialLoss', 'AerialWin%', 'G+APer90', 'GlsxxPer90', 'GoalsPer90', 'ShotsPer90', 'SoTPer90', 'SoT%Per90', 'Sh/90Per90', 'SoT/90Per90', 'G/ShPer90', 'G/SoTPer90', 'AvgShotDistancePer90', 'FKShotsPer90', 'PKPer90', 'PKsAttPer90', 'xGPer90', 'npxGPer90', 'npxG/ShPer90', 'G-xGPer90', 'npG-xGPer90', 'PassesCompletedPer90', 'PassesAttemptedPer90', 'TotCmp%Per90', 'TotalPassDistPer90', 'ProgPassDistPer90', 'ShortPassCmpPer90', 'ShortPassAttPer90', 'ShortPassCmp%Per90', 'MedPassCmpPer90', 'MedPassAttPer90', 'MedPassCmp%Per90', 'LongPassCmpPer90', 'LongPassAttPer90', 'LongPassCmp%Per90', 'AssistsPer90', 'xAGPer90', 'xAPer90', 'A-xAGPer90', 'KeyPassesPer90', 'Final1/3CmpPer90', 'PenAreaCmpPer90', 'CrsPenAreaCmpPer90', 'ProgPassesPer90', 'LivePassPer90', 'DeadPassPer90', 'FKPassesPer90', 'ThruBallsPer90', 'SwitchesPer90', 'CrsPer90', 'ThrowInPer90', 'CKPer90', 'InSwingCKPer90', 'OutSwingCKPer90', 'StrCKPer90', 'CmpxxxPer90', 'PassesToOffPer90', 'PassesBlockedPer90', 'SCAPer90', 'SCA90Per90', 'SCAPassLivePer90', 'SCAPassDeadPer90', 'SCADribPer90', 'SCAShPer90', 'SCAFldPer90', 'SCADefPer90', 'GCAPer90', 'GCA90Per90', 'GCAPassLivePer90', 'GCAPassDeadPer90', 'GCADribPer90', 'GCAShPer90', 'GCAFldPer90', 'GCADefPer90', 'TklPer90', 'TklWinPossPer90', 'Def3rdTklPer90', 'Mid3rdTklPer90', 'Att3rdTklPer90', 'DrbTklPer90', 'DrbPastAttPer90', 'DrbTkl%Per90', 'DrbPastPer90', 'BlocksPer90', 'ShBlocksPer90', 'PassBlocksPer90', 'IntPer90', 'Tkl+IntPer90', 'ClrPer90', 'ErrPer90', 'TouchesPer90', 'DefPenTouchPer90', 'Def3rdTouchPer90', 'Mid3rdTouchPer90', 'Att3rdTouchPer90', 'AttPenTouchPer90', 'LiveTouchPer90', 'AttDrbPer90', 'SuccDrbPer90', 'DrbSucc%Per90', 'TimesTackledPer90', 'TimesTackled%Per90', 'CarriesPer90', 'TotalCarryDistancePer90', 'ProgCarryDistancePer90', 'ProgCarriesPer90', 'CarriesToFinalThirdPer90', 'CarriesToPenAreaPer90', 'CarryMistakesPer90', 'DisposesedPer90', 'ReceivedPassPer90', 'ProgPassesRecPer90', 'YellowsPer90', 'RedsPer90', 'Yellow2Per90', 'FlsPer90', 'FldPer90', 'OffPer90', 'PKwonPer90', 'PKconPer90', 'OGPer90', 'RecovPer90', 'AerialWinsPer90', 'AerialLossPer90', 'AerialWin%Per90', '90sPer90', 'AvgTeamPoss', 'OppTouches', 'TeamMins', 'TeamTouches90', 'pAdjTkl+IntPer90', 'pAdjClrPer90', 'pAdjShBlocksPer90', 'pAdjPassBlocksPer90', 'pAdjIntPer90', 'pAdjDrbTklPer90', 'pAdjTklWinPossPer90', 'pAdjDrbPastPer90', 'pAdjAerialWinsPer90', 'pAdjAerialLossPer90', 'pAdjDrbPastAttPer90', 'TouchCentrality', 'Tkl+IntPer600OppTouch', 'pAdjTouchesPer90', 'CarriesPer50Touches', 'ProgCarriesPer50Touches', 'ProgPassesPer50CmpPasses', 'ProgDistancePerCarry', 'ProgCarryEfficiency', 'PlayerFBref', 'ShortPass%', 'MediumPass%', 'LongPass%', 'ProgPass%', 'Switch%', 'KeyPass%', 'Final3rdPass%', 'ThroughPass%', 'Def3rdTouch%', 'Mid3rdTouch%', 'Att3rdTouch%', 'AttPenTouch%', 'ActionsPerTouch', 'Def3rdTkl%', 'Mid3rdTkl%', 'Att3rdTkl%', 'LineBreakingPassesPer90', 'LineBreakingPass%', 'LooseBallWinsPer90'
    ]

    for metric in metrics_to_rank:
        # Create the percentile column name
        percentile_col = f'{metric}_PR'
        
        # Calculate percentile rank
        df[percentile_col] = df.groupby(['Season', 'Extract'])[metric].rank(pct=True) * 100
        
        # Round to 1 decimal place
        df[percentile_col] = df[percentile_col].round(1)

    # Drop all duplicates
    df = df.sort_values('Min', ascending=False).drop_duplicates(subset=['Player Name', 'Squad', 'Season', 'Main Position'], keep='first')

    return df

def create_percentile_rankings_filtered(fbref_position, position_template, df):

    metrics_to_rank = [
    'Min', 'G+A', 'Glsxx', 'Goals', 'Shots', 'SoT', 'SoT%', 'Sh/90', 'SoT/90', 'G/Sh', 'G/SoT', 'AvgShotDistance', 'FKShots', 'PK', 'PKsAtt', 'xG', 'npxG', 'npxG/Sh', 'G-xG', 'npG-xG', 'PassesCompleted', 'PassesAttempted', 'TotCmp%', 'TotalPassDist', 'ProgPassDist', 'ShortPassCmp', 'ShortPassAtt', 'ShortPassCmp%', 'MedPassCmp', 'MedPassAtt', 'MedPassCmp%', 'LongPassCmp', 'LongPassAtt', 'LongPassCmp%', 'Assists', 'xAG', 'xA', 'A-xAG', 'KeyPasses', 'Final1/3Cmp', 'PenAreaCmp', 'CrsPenAreaCmp', 'ProgPasses', 'LivePass', 'DeadPass', 'FKPasses', 'ThruBalls', 'Switches', 'Crs', 'ThrowIn', 'CK', 'InSwingCK', 'OutSwingCK', 'StrCK', 'Cmpxxx', 'PassesToOff', 'PassesBlocked', 'SCA', 'SCA90', 'SCAPassLive', 'SCAPassDead', 'SCADrib', 'SCASh', 'SCAFld', 'SCADef', 'GCA', 'GCA90', 'GCAPassLive', 'GCAPassDead', 'GCADrib', 'GCASh', 'GCAFld', 'GCADef', 'Tkl', 'TklWinPoss', 'Def3rdTkl', 'Mid3rdTkl', 'Att3rdTkl', 'DrbTkl', 'DrbPastAtt', 'DrbTkl%', 'DrbPast', 'Blocks', 'ShBlocks', 'PassBlocks', 'Int', 'Tkl+Int', 'Clr', 'Err', 'Touches', 'DefPenTouch', 'Def3rdTouch', 'Mid3rdTouch', 'Att3rdTouch', 'AttPenTouch', 'LiveTouch', 'AttDrb', 'SuccDrb', 'DrbSucc%', 'TimesTackled', 'TimesTackled%', 'Carries', 'TotalCarryDistance', 'ProgCarryDistance', 'ProgCarries', 'CarriesToFinalThird', 'CarriesToPenArea', 'CarryMistakes', 'Disposesed', 'ReceivedPass', 'ProgPassesRec', 'Yellows', 'Reds', 'Yellow2', 'Fls', 'Fld', 'Off', 'PKwon', 'PKcon', 'OG', 'Recov', 'AerialWins', 'AerialLoss', 'AerialWin%', 'G+APer90', 'GlsxxPer90', 'GoalsPer90', 'ShotsPer90', 'SoTPer90', 'SoT%Per90', 'Sh/90Per90', 'SoT/90Per90', 'G/ShPer90', 'G/SoTPer90', 'AvgShotDistancePer90', 'FKShotsPer90', 'PKPer90', 'PKsAttPer90', 'xGPer90', 'npxGPer90', 'npxG/ShPer90', 'G-xGPer90', 'npG-xGPer90', 'PassesCompletedPer90', 'PassesAttemptedPer90', 'TotCmp%Per90', 'TotalPassDistPer90', 'ProgPassDistPer90', 'ShortPassCmpPer90', 'ShortPassAttPer90', 'ShortPassCmp%Per90', 'MedPassCmpPer90', 'MedPassAttPer90', 'MedPassCmp%Per90', 'LongPassCmpPer90', 'LongPassAttPer90', 'LongPassCmp%Per90', 'AssistsPer90', 'xAGPer90', 'xAPer90', 'A-xAGPer90', 'KeyPassesPer90', 'Final1/3CmpPer90', 'PenAreaCmpPer90', 'CrsPenAreaCmpPer90', 'ProgPassesPer90', 'LivePassPer90', 'DeadPassPer90', 'FKPassesPer90', 'ThruBallsPer90', 'SwitchesPer90', 'CrsPer90', 'ThrowInPer90', 'CKPer90', 'InSwingCKPer90', 'OutSwingCKPer90', 'StrCKPer90', 'CmpxxxPer90', 'PassesToOffPer90', 'PassesBlockedPer90', 'SCAPer90', 'SCA90Per90', 'SCAPassLivePer90', 'SCAPassDeadPer90', 'SCADribPer90', 'SCAShPer90', 'SCAFldPer90', 'SCADefPer90', 'GCAPer90', 'GCA90Per90', 'GCAPassLivePer90', 'GCAPassDeadPer90', 'GCADribPer90', 'GCAShPer90', 'GCAFldPer90', 'GCADefPer90', 'TklPer90', 'TklWinPossPer90', 'Def3rdTklPer90', 'Mid3rdTklPer90', 'Att3rdTklPer90', 'DrbTklPer90', 'DrbPastAttPer90', 'DrbTkl%Per90', 'DrbPastPer90', 'BlocksPer90', 'ShBlocksPer90', 'PassBlocksPer90', 'IntPer90', 'Tkl+IntPer90', 'ClrPer90', 'ErrPer90', 'TouchesPer90', 'DefPenTouchPer90', 'Def3rdTouchPer90', 'Mid3rdTouchPer90', 'Att3rdTouchPer90', 'AttPenTouchPer90', 'LiveTouchPer90', 'AttDrbPer90', 'SuccDrbPer90', 'DrbSucc%Per90', 'TimesTackledPer90', 'TimesTackled%Per90', 'CarriesPer90', 'TotalCarryDistancePer90', 'ProgCarryDistancePer90', 'ProgCarriesPer90', 'CarriesToFinalThirdPer90', 'CarriesToPenAreaPer90', 'CarryMistakesPer90', 'DisposesedPer90', 'ReceivedPassPer90', 'ProgPassesRecPer90', 'YellowsPer90', 'RedsPer90', 'Yellow2Per90', 'FlsPer90', 'FldPer90', 'OffPer90', 'PKwonPer90', 'PKconPer90', 'OGPer90', 'RecovPer90', 'AerialWinsPer90', 'AerialLossPer90', 'AerialWin%Per90', '90sPer90', 'AvgTeamPoss', 'OppTouches', 'TeamMins', 'TeamTouches90', 'pAdjTkl+IntPer90', 'pAdjClrPer90', 'pAdjShBlocksPer90', 'pAdjPassBlocksPer90', 'pAdjIntPer90', 'pAdjDrbTklPer90', 'pAdjTklWinPossPer90', 'pAdjDrbPastPer90', 'pAdjAerialWinsPer90', 'pAdjAerialLossPer90', 'pAdjDrbPastAttPer90', 'TouchCentrality', 'Tkl+IntPer600OppTouch', 'pAdjTouchesPer90', 'CarriesPer50Touches', 'ProgCarriesPer50Touches', 'ProgPassesPer50CmpPasses', 'ProgDistancePerCarry', 'ProgCarryEfficiency', 'PlayerFBref', 'ShortPass%', 'MediumPass%', 'LongPass%', 'ProgPass%', 'Switch%', 'KeyPass%', 'Final3rdPass%', 'ThroughPass%', 'Def3rdTouch%', 'Mid3rdTouch%', 'Att3rdTouch%', 'AttPenTouch%', 'ActionsPerTouch', 'Def3rdTkl%', 'Mid3rdTkl%', 'Att3rdTkl%', 'LineBreakingPassesPer90', 'LineBreakingPass%', 'LooseBallWinsPer90'
    ]

    df = df[df['Pos'].isin(fbref_position)]

    mask_null = df['Position Group'].isnull()

    if position_template == 'ST':# and 'FW' in fbref_position:
        df.loc[mask_null, 'Position Group'] = 'ST'

    elif position_template == 'W':# and 'FW' in fbref_position:
        df.loc[mask_null, 'Position Group'] = 'W'

    elif position_template == 'AM':# and 'MF' in fbref_position:
        df.loc[mask_null, 'Position Group'] = 'AM'

    elif position_template == 'CM':# and 'MF' in fbref_position:
        df.loc[mask_null, 'Position Group'] = 'CM'

    elif position_template == 'DM':# and 'MF' in fbref_position:
        df.loc[mask_null, 'Position Group'] = 'DM'

    elif position_template == 'CB':# and 'DF' in fbref_position:
        df.loc[mask_null, 'Position Group'] = 'CB'

    elif position_template == 'FB':# and 'DF' in fbref_position:
        df.loc[mask_null, 'Position Group'] = 'FB'


    #df = df[df['Position Group'].notnull()]

    for metric in metrics_to_rank:
        # Create the percentile column name
        percentile_col = f'{metric}_PR'
        
        # Calculate percentile rank
        df[percentile_col] = df.groupby(['Position Group', 'Season', 'Extract'])[metric].rank(pct=True) * 100
        
        # Round to 1 decimal place
        df[percentile_col] = df[percentile_col].round(1)

    # Drop all duplicates
    df = df.sort_values('Min', ascending=False).drop_duplicates(subset=['Player', 'Squad', 'Season', 'Position Group'], keep='first')

    return df

def create_aggregated_columns(df):

    # All of these are aggregated columns
    df['Aerial Ability']= ((df['AerialWin%_PR']*0.8)+(df['pAdjAerialWinsPer90_PR']*0.2))
    df['Box Defending']= (df['ShBlocksPer90_PR']+df['ClrPer90_PR']+df['pAdjClrPer90_PR']+df['pAdjShBlocksPer90_PR'])/4
    df['1v1 Defending']= ((df['DrbTklPer90_PR']*0.25)+(df['DrbTkl%_PR']*0.75))
    df['Defensive Awareness']= (df['IntPer90_PR']+df['LooseBallWinsPer90_PR']+df['pAdjIntPer90_PR'])/3
    df['Pass Progression'] = (df['ProgPassesPer90_PR']+df['ProgPass%_PR']+df['LineBreakingPassesPer90_PR']+df['LineBreakingPass%_PR'])/4
    df['Pass Retention'] = (df['TotCmp%_PR'])
    df['Ball Carrying'] = (df['ProgCarryDistancePer90_PR']+df['ProgCarriesPer90_PR']+df['ProgCarriesPer50Touches_PR'])/3
    df['Volume of Take-ons'] = (df['AttDrbPer90_PR'])
    df['Retention from Take-ons'] = (df['DrbSucc%_PR'])
    df['Chance Creation'] = (df['xAGPer90_PR']+df['xAPer90_PR']+df['KeyPassesPer90_PR']+df['SCAPassLivePer90_PR'])/4
    df['Impact in and around box'] = (df['Final1/3CmpPer90_PR']+df['PenAreaCmpPer90_PR']+df['CrsPenAreaCmpPer90_PR']+df['ThruBallsPer90_PR']+df['Att3rdTouchPer90_PR']+df['AttPenTouchPer90_PR'])/6
    df['Shot Volume'] = (df['ShotsPer90_PR'])
    df['Shot Quality'] = (df['npxG/Sh_PR'])
    df['Self-created Shots'] = (df['SCADribPer90_PR'])
    df['Switching Play'] = (df['SwitchesPer90_PR'])
    df['Defensive Intensity'] = (df['Tkl+IntPer90_PR']+df['Tkl+IntPer600OppTouch_PR']+df['TklWinPossPer90_PR']+df['FlsPer90_PR']+df['PassBlocksPer90_PR']+df['LooseBallWinsPer90_PR'])/6

    # Traits
    df['Attempts a lot of dribbles'] = np.where(df['AttDrbPer90_PR'] >= 75, 1, 0)
    df['Carries the ball frequently'] = np.where(df['ProgCarriesPer50Touches_PR'] >= 75, 1, 0)
    df['Creates a lot of his own shots'] = np.where(df['SCADribPer90_PR'] >= 75, 1, 0)
    df['Shoots frequently'] = np.where(df['ShotsPer90_PR'] >= 75, 1, 0)
    df['Attempts a lot of through balls'] = np.where(df['ThroughPass%_PR'] >= 75, 1, 0)
    df['Gets fouled frequently'] = np.where(df['FldPer90_PR'] >= 75, 1, 0)
    df['Switches the ball frequently'] = np.where(df['Switch%_PR'] >= 75, 1, 0)
    df['Makes a lot of tackles'] = np.where(df['TklPer90_PR'] >= 75, 1, 0)
    df['Plays a lot of progressive passes'] = np.where(df['ProgPass%_PR'] >= 75, 1, 0)
    df['Plays a lot of short passes'] = np.where(df['ShortPass%_PR'] >= 75, 1, 0)
    df['Plays a lot of long passes'] = np.where(df['LongPass%_PR'] >= 75, 1, 0)
    if df['Position Group'].any() != 'ST':
        df['Plays a lot of line-breaking passes'] = np.where((df['LineBreakingPass%_PR'] >= 75) & (df['PassesCompletedPer90_PR'] >= 25), 1, 0)
    df['Has a high share of teams total touches'] = np.where(df['TouchCentrality_PR'] >= 75, 1, 0)
    df['Has a high share of touches in defensive 3rd'] = np.where(df['Def3rdTouch%_PR'] >= 75, 1, 0)
    df['Has a high share of touches in middle 3rd'] = np.where(df['Mid3rdTouch%_PR'] >= 75, 1, 0)
    df['Has a high share of touches in final 3rd'] = np.where(df['Att3rdTouch%_PR'] >= 75, 1, 0)
    df['Has a high share of touches in the penalty box'] = np.where(df['AttPenTouch%_PR'] >= 75, 1, 0)
    df['Competes in a lot of aerial duels'] = np.where(df['AerialWinsPer90_PR'] >= 75, 1, 0)
    df['Fouls frequently'] = np.where(df['FlsPer90_PR'] >= 75, 1, 0)
    df['Receives a lot of progressive passes'] = np.where(df['ProgPassesRecPer90_PR'] >= 75, 1, 0)
    df['Has a high share of carries into dangerous areas'] = np.where(df['ProgCarryEfficiency_PR'] >= 75, 1, 0)
    df['Crosses the ball frequently'] = np.where(df['CrsPer90_PR'] >= 75, 1, 0)
    df['Shoots from poor areas'] = np.where(df['npxG/Sh_PR'] <= 25, 1, 0)
    df['Shoots from good areas'] = np.where(df['npxG/Sh_PR'] >= 75, 1, 0)
    df['Carries the ball over long distances'] = np.where(df['ProgDistancePerCarry_PR'] >= 75, 1, 0)
    df['Has a high share of tackles in defensive 3rd'] = np.where(df['Def3rdTkl%_PR'] >= 75, 1, 0)
    df['Has a high share of tackles in middle 3rd'] = np.where(df['Mid3rdTkl%_PR'] >= 75, 1, 0)
    df['Has a high share of tackles in final 3rd'] = np.where(df['Att3rdTkl%_PR'] >= 75, 1, 0)
    df['Intercepts passes frequently'] = np.where(df['IntPer90_PR'] >= 75, 1, 0)
    df['Sweeps up loose balls frequently'] = np.where(df['LooseBallWinsPer90_PR'] >= 75, 1, 0)
    df['Blocks passes frequently'] = np.where(df['PassBlocksPer90_PR'] >= 75, 1, 0)
    df['Takes few touches per action'] = np.where(df['ActionsPerTouch_PR'] >= 75, 1, 0)
    df['Takes a lot of touches per action'] = np.where(df['ActionsPerTouch_PR'] <= 25, 1, 0)
    #if df['Position Group'].any() != 'ST':
        #df['Plays a lot of line-breaking passes'] = np.where(
            #((df['ShortPass%_PR'] > 70) | (df['MediumPass%_PR'] > 70)) & 
            #(df['ProgPass%_PR'] > 70) & 
            #(df['PassesCompletedPer90_PR'] > 40) & 
            #(df['ProgPassesPer90_PR'] > 50), 1, 0)

    #Strengths
    df['Aerial Duels'] = np.where(df['AerialWin%_PR'] >= 75, 1, 0)
    df['Tackling'] = np.where(df['DrbTkl%_PR'] >= 75, 1, 0)
    df['Passing Completion'] = np.where(df['TotCmp%_PR'] >= 75, 1, 0)
    df['Long Passing Completion'] = np.where(df['LongPassCmp%_PR'] >= 75, 1, 0)
    df['Creating Chances'] = np.where(df['xAPer90_PR'] >= 75, 1, 0)
    df['Dribbling'] = np.where(df['DrbSucc%_PR'] >= 75, 1, 0)
    df['Finishing'] = np.where(df['npG-xGPer90_PR'] >= 75, 1, 0)
    df['Shooting on Target'] = np.where(df['SoT%_PR'] >= 75, 1, 0)

    #Weaknesses
    df['Aerial Duels '] = np.where(df['AerialWin%_PR'] <= 25, 1, 0)
    df['Tackling '] = np.where(df['DrbTkl%_PR'] <= 25, 1, 0)
    df['Passing Completion '] = np.where(df['TotCmp%_PR'] <= 25, 1, 0)
    df['Long Passing Completion '] = np.where(df['LongPassCmp%_PR'] <= 25, 1, 0)
    df['Creating Chances '] = np.where(df['xAPer90_PR'] <= 25, 1, 0)
    df['Dribbling '] = np.where(df['DrbSucc%_PR'] <= 25, 1, 0)
    df['Finishing '] = np.where(df['npG-xGPer90_PR'] <= 25, 1, 0)
    df['Shooting on Target '] = np.where(df['SoT%_PR'] <= 25, 1, 0)

    #Positional Rankings
    df['Ball Playing CB'] = ((df['PassesCompletedPer90_PR']*0.15)+(df['ProgPassesPer90_PR']*0.175)+(df['ShortPass%_PR']*0.10)+(df['ProgPass%_PR']*0.225)+(df['Switch%_PR']*0.15)+(df['ProgCarriesPer90_PR']*0.20))
    df['Sweeper CB'] = ((df['IntPer90_PR']*0.30)+(df['RecovPer90_PR']*0.25)+(df['AerialWin%_PR']*0.20)+(df['Def3rdTkl%_PR']*0.20)+(df['DrbTkl%_PR']*0.05))
    df['Aggressor'] = ((df['TklPer90_PR']*0.15)+(df['DrbTkl%_PR']*0.125)+(df['FlsPer90_PR']*0.125)+(df['Mid3rdTkl%_PR']*0.15)+(df['Att3rdTkl%_PR']*0.05)+(df['PassBlocksPer90_PR']*0.15)+(df['AerialWin%_PR']*0.125)+(df['AerialWinsPer90_PR']*0.125))
    df['Wide CB'] = ((df['TklPer90_PR']*0.25)+(df['DrbTkl%_PR']*0.25)+(df['ProgPassesPer90_PR']*0.15)+(df['ProgCarriesPer90_PR']*0.15)+(df['AerialWin%_PR']*0.10)+(df['CarriesToFinalThirdPer90_PR']*0.10))
    df['Box Defender'] = ((df['Def3rdTkl%_PR']*0.15)+(df['AerialWinsPer90_PR']*0.175)+(df['AerialWin%_PR']*0.175)+(df['pAdjClrPer90_PR']*0.25)+(df['pAdjShBlocksPer90_PR']*0.25))     
    df['Defensive FB'] = ((df['Def3rdTouch%_PR']*0.10)+(df['Def3rdTkl%_PR']*0.15)+(df['TklPer90_PR']*0.20)+(df['DrbTkl%_PR']*0.25)+(df['AerialWin%_PR']*0.10)+(df['pAdjClrPer90_PR']*0.10)+(df['pAdjShBlocksPer90_PR']*0.10))
    df['Progressive FB'] = ((df['TouchCentrality_PR']*0.10)+(df['Mid3rdTouch%_PR']*0.10)+(df['ProgPass%_PR']*0.15)+(df['LineBreakingPass%_PR']*0.15)+(df['ProgCarriesPer90_PR']*0.10)+(df['AttDrbPer90_PR']*0.10)+(df['DrbSucc%_PR']*0.10)+(df['PenAreaCmpPer90_PR']*0.10)+(df['ThroughPass%_PR']*0.10)) 
    df['All-Rounder FB'] = ((df['Aerial Ability']+df['Defensive Intensity']+df['1v1 Defending']+df['Pass Progression']+df['Ball Carrying']+df['Volume of Take-ons']+df['Chance Creation']+df['Shot Volume'])/8) 
    df['Wing Back'] =  ((df['ShotsPer90_PR']*0.0727272727)+(df['Att3rdTouch%_PR']*0.0727272727)+(df['AttPenTouch%_PR']*0.0727272727)+(df['TklPer90_PR']*0.0727272727)+(df['AerialWin%_PR']*0.0727272727)+(df['Mid3rdTkl%_PR']*0.0727272727)+(df['Att3rdTkl%_PR']*0.0727272727)+(df['ProgPassesRecPer90_PR']*0.0727272727)+(df['AttDrbPer90_PR']*0.0727272727)+(df['ProgCarriesPer90_PR']*0.0727272727)+(df['ProgDistancePerCarry_PR']*0.0727272727)+(df['xAPer90_PR']*0.10)+(df['CrsPer90_PR']*0.10)) 
    df['Deep Lying DM'] = ((df['TouchCentrality_PR']*0.10)+(df['ProgPass%_PR']*0.15)+(df['LineBreakingPass%_PR']*0.15)+(df['Switch%_PR']*0.10)+(df['PassesCompletedPer90_PR']*0.10)+(df['TotCmp%_PR']*0.10)+(df['Def3rdTouch%_PR']*0.075)+(df['AttDrbPer90_PR']*0.075)+(df['DrbSucc%_PR']*0.075)+(df['ThroughPass%_PR']*0.075)) 
    df['Ball Winning DM'] = ((df['Mid3rdTkl%_PR']*0.15)+(df['TklPer90_PR']*0.25)+(df['DrbTkl%_PR']*0.20)+(df['PassBlocksPer90_PR']*0.20)+(df['AerialWinsPer90_PR']*0.10)+(df['RecovPer90_PR']*0.10)) 
    df['Sweeper DM'] = ((df['IntPer90_PR']*0.25)+(df['RecovPer90_PR']*0.10)+(df['AerialWinsPer90_PR']*0.15)+(df['AerialWin%_PR']*0.15)+(df['TotCmp%_PR']*0.10)+(df['LooseBallWinsPer90_PR']*0.25))
    df['Ball Carrying DM'] = ((df['AttDrbPer90_PR']*0.25)+(df['DrbSucc%_PR']*0.25)+(df['ProgCarriesPer90_PR']*0.15)+(df['FldPer90_PR']*0.10)+(df['ReceivedPassPer90_PR']*0.125)+(df['TouchCentrality_PR']*0.125))
    df['Box-to-Box'] = ((df['ShotsPer90_PR']*0.20)+(df['npxG/Sh_PR']*0.10)+(df['TklPer90_PR']*0.15)+(df['LooseBallWinsPer90_PR']*0.15)+(df['KeyPassesPer90_PR']*0.10)+(df['ProgCarriesPer90_PR']*0.10)+(df['AerialWinsPer90_PR']*0.10)+(df['ProgPassesRecPer90_PR']*0.10))
    df['Deep Lying CM'] = ((df['TouchCentrality_PR']*0.10)+(df['ProgPass%_PR']*0.15)+(df['LineBreakingPass%_PR']*0.15)+(df['Switch%_PR']*0.10)+(df['PassesCompletedPer90_PR']*0.10)+(df['TotCmp%_PR']*0.10)+(df['KeyPass%_PR']*0.075)+(df['AttDrbPer90_PR']*0.075)+(df['DrbSucc%_PR']*0.075)+(df['ThroughPass%_PR']*0.075))  
    df['Ball Winning CM'] = ((df['Att3rdTkl%_PR']*0.15)+(df['TklPer90_PR']*0.25)+(df['DrbTkl%_PR']*0.20)+(df['PassBlocksPer90_PR']*0.20)+(df['AerialWinsPer90_PR']*0.10)+(df['RecovPer90_PR']*0.10))     
    df['Sweeper CM'] = ((df['IntPer90_PR']*0.25)+(df['RecovPer90_PR']*0.10)+(df['AerialWinsPer90_PR']*0.15)+(df['AerialWin%_PR']*0.15)+(df['TotCmp%_PR']*0.10)+(df['LooseBallWinsPer90_PR']*0.25))
    df['Ball Carrying CM'] = ((df['AttDrbPer90_PR']*0.25)+(df['DrbSucc%_PR']*0.25)+(df['ProgCarriesPer90_PR']*0.15)+(df['FldPer90_PR']*0.10)+(df['ReceivedPassPer90_PR']*0.125)+(df['TouchCentrality_PR']*0.125))
    df['All-Rounder CM'] = ((df['ShotsPer90_PR']*0.20)+(df['npxG/Sh_PR']*0.10)+(df['TklPer90_PR']*0.15)+(df['RecovPer90_PR']*0.15)+(df['KeyPassesPer90_PR']*0.10)+(df['ProgCarriesPer90_PR']*0.10)+(df['AerialWinsPer90_PR']*0.10)+(df['ProgPassesRecPer90_PR']*0.10)) 
    df['Playmaking CM'] = ((df['ThroughPass%_PR']*0.20)+(df['KeyPass%_PR']*0.15)+(df['ProgPassesRecPer90_PR']*0.15)+(df['xAPer90_PR']*0.20)+(df['PenAreaCmpPer90_PR']*0.10)+(df['ShotsPer90_PR']*0.10)+(df['SCAPer90_PR']*0.10))
    df['Box Crasher'] = ((df['AerialWinsPer90_PR']*0.125)+(df['AerialWin%_PR']*0.125)+(df['ShotsPer90_PR']*0.25)+(df['npxGPer90_PR']*0.20)+(df['npxG/Sh_PR']*0.15)+(df['AttPenTouch%_PR']*0.15))
    df['Playmaking 10'] = ((df['ThroughPass%_PR']*0.15)+(df['KeyPass%_PR']*0.15)+(df['ProgPassesRecPer90_PR']*0.10)+(df['AttDrbPer90_PR']*0.10)+(df['ProgCarryEfficiency_PR']*0.10)+(df['xAPer90_PR']*0.15)+(df['PenAreaCmpPer90_PR']*0.10)+(df['ShotsPer90_PR']*0.10)+(df['SCAPer90_PR']*0.05))
    df['Second Striker'] = ((df['ShotsPer90_PR']*0.20)+(df['npxGPer90_PR']*0.20)+(df['npxG/Sh_PR']*0.15)+(df['AttPenTouch%_PR']*0.10)+(df['SCADribPer90_PR']*0.10)+(df['ProgDistancePerCarry_PR']*0.10)+(df['ProgPassesRecPer90_PR']*0.10)+(df['LooseBallWinsPer90_PR']*0.05))
    df['Ball Carrying AM'] = ((df['AttDrbPer90_PR']*0.225)+(df['DrbSucc%_PR']*0.225)+(df['ProgCarriesPer90_PR']*0.15)+(df['FldPer90_PR']*0.075)+(df['ProgPassesRecPer90_PR']*0.125)+(df['SCADribPer90_PR']*0.20))
    df['Touchline Winger'] = ((df['ProgCarryEfficiency_PR']*0.15)+(df['ProgPassesRecPer90_PR']*0.10)+(df['AttDrbPer90_PR']*0.20)+(df['xAPer90_PR']*0.15)+(df['PenAreaCmpPer90_PR']*0.15)+(df['CrsPer90_PR']*0.15)+(df['Att3rdTouch%_PR']*0.10))
    df['Inside Forward'] = ((df['ProgPassesRecPer90_PR']*0.125)+(df['ShotsPer90_PR']*0.175)+(df['AttPenTouch%_PR']*0.125)+(df['ProgCarryEfficiency_PR']*0.10)+(df['ProgDistancePerCarry_PR']*0.10)+(df['npxGPer90_PR']*0.175)+(df['SCADribPer90_PR']*0.15)+(df['AttDrbPer90_PR']*0.05))
    df['Playmaking Winger'] = ((df['ThroughPass%_PR']*0.15)+(df['KeyPass%_PR']*0.15)+(df['ProgPassesRecPer90_PR']*0.10)+(df['AttDrbPer90_PR']*0.10)+(df['ProgCarryEfficiency_PR']*0.10)+(df['xAPer90_PR']*0.15)+(df['PenAreaCmpPer90_PR']*0.10)+(df['ShotsPer90_PR']*0.10)+(df['SCAPer90_PR']*0.05))
    df['Outlet Winger'] = ((df['AerialWinsPer90_PR']*0.15)+(df['AerialWin%_PR']*0.15)+(df['AttDrbPer90_PR']*0.20)+(df['ProgDistancePerCarry_PR']*0.15)+(df['ProgCarryEfficiency_PR']*0.15)+(df['ShotsPer90_PR']*0.10)+(df['ProgPassesRecPer90_PR']*0.05)+(df['Def3rdTkl%_PR']*0.05))
    df['Outlet'] = ((df['ProgDistancePerCarry_PR']*0.125)+(df['ProgCarryEfficiency_PR']*0.10)+(df['AttDrbPer90_PR']*0.125)+(df['DrbSucc%_PR']*0.10)+(df['AerialWinsPer90_PR']*0.125)+(df['AerialWin%_PR']*0.1375)+(df['ShotsPer90_PR']*0.10)+(df['SCADribPer90_PR']*0.1375)+(df['FldPer90_PR']*0.05))
    df['Target Man'] = ((df['AerialWin%_PR']*0.15)+(df['AerialWinsPer90_PR']*0.175)+(df['KeyPass%_PR']*0.10)+(df['ShortPass%_PR']*0.05)+(df['npxGPer90_PR']*0.20)+(df['npxG/Sh_PR']*0.20)+(df['AttPenTouch%_PR']*0.125))
    df['Poacher'] = ((df['npxGPer90_PR']*0.25)+(df['npxG/Sh_PR']*0.25)+(df['AttPenTouch%_PR']*0.25)+(df['ShotsPer90_PR']*0.25))
    df['False 9'] = ((df['TouchCentrality_PR']*0.10)+(df['Mid3rdTouch%_PR']*0.05)+(df['AttDrbPer90_PR']*0.15)+(df['DrbSucc%_PR']*0.15)+(df['ProgCarryEfficiency_PR']*0.10)+(df['ThroughPass%_PR']*0.10)+(df['xAPer90_PR']*0.10)+(df['ShotsPer90_PR']*0.15)+(df['SCADribPer90_PR']*0.10))  

    return df

font_normal = FontManager('https://raw.githubusercontent.com/googlefonts/roboto/main/'
                          'src/hinted/Roboto-Regular.ttf')
font_italic = FontManager('https://raw.githubusercontent.com/googlefonts/roboto/main/'
                          'src/hinted/Roboto-Italic.ttf')
font_bold = FontManager('https://raw.githubusercontent.com/google/fonts/main/apache/robotoslab/'
                        'RobotoSlab[wght].ttf')

# Define metrics for each position group (same as radar chart)
metrics_by_position = {
    'FB': [
        'Aerial Ability', 'Defensive Awareness', 'Defensive Intensity', '1v1 Defending', 
        'Pass Progression', 'Pass Retention', 'Ball Carrying',
        'Volume of Take-ons', 'Chance Creation', 'Impact in and around box'
    ],
    'CB': [
        'Aerial Ability', 'Box Defending', 'Defensive Awareness', '1v1 Defending',
        'Defensive Intensity', 'Pass Progression', 'Pass Retention',
        'Switching Play', 'Ball Carrying', 'Shot Volume'
    ],
    'DM': [
        'Aerial Ability', 'Defensive Awareness', 'Defensive Intensity', '1v1 Defending',
        'Pass Progression', 'Pass Retention', 'Switching Play', 'Volume of Take-ons',
        'Retention from Take-ons', 'Shot Volume'
    ],
    'CM': [
        'Defensive Awareness', 'Defensive Intensity', 'Pass Progression', 'Pass Retention',
        'Ball Carrying', 'Volume of Take-ons', 'Retention from Take-ons',
        'Chance Creation', 'Impact in and around box', 'Shot Volume'
    ],
    'AM': [
        'Defensive Intensity', 'Pass Progression', 'Pass Retention',
        'Volume of Take-ons', 'Retention from Take-ons', 'Chance Creation', 'Impact in and around box',
        'Shot Volume', 'Shot Quality', 'Self-created Shots'
    ],
    'W': [
        'Defensive Intensity', 'Pass Retention',
        'Ball Carrying', 'Volume of Take-ons', 'Retention from Take-ons',
        'Chance Creation', 'Impact in and around box',
        'Shot Volume', 'Shot Quality', 'Self-created Shots'
    ],
    'ST': [
        'Aerial Ability', 'Defensive Intensity',
        'Pass Retention', 'Ball Carrying', 'Volume of Take-ons',
        'Chance Creation', 'Impact in and around box',
        'Shot Volume', 'Shot Quality', 'Self-created Shots'
    ]
}

metrics_by_area = {
    'Defensive': [
        'Aerial Ability', 'Defensive Awareness', 'Defensive Intensity', '1v1 Defending', 'Box Defending'
    ],
    'Passing': [
        'Pass Progression', 'Pass Retention', 'Switching Play'
    ],
    'Ball Carrying': [
        'Ball Carrying', 'Volume of Take-ons', 'Retention from Take-ons'
    ],
    'Chance Creation': [
        'Chance Creation', 'Impact in and around box'
    ],
    'Shooting': [
        'Shot Volume', 'Shot Quality', 'Self-created Shots'
    ]
}

def create_player_pizza(player_name, season_name, position_group, df, save_fig=False):
    """
    Creates a pizza chart for a specified player based on their position group
    
    Args:
    player_name (str): Name of the player
    df (DataFrame): DataFrame containing player data (default: df_combined)
    save_fig (bool): Whether to save the figure (default: False)
    """
    
    # Get player's data and position group
    player_data = df[(df['Player Name'] == player_name) & (df['Season'] == season_name)].iloc[0]
    squad = player_data['Squad']
    season_filter = player_data['Season']
    extract = player_data['Extract']

    color_mapping = {
        'Defensive': '#1A78CF',
        'Passing': '#008000',
        'Ball Carrying': '#FFA500',
        'Chance Creation': '#800080',
        'Shooting': '#FF0000'
    }
    
    # Get metrics for player's position
    metrics = metrics_by_position[position_group]
    values = [round(player_data[metric]) for metric in metrics]
    
    def get_colors_for_position_group(position_group):
        metrics = metrics_by_position.get(position_group, [])
        slice_colors = []

        for metric in metrics:
            found = False
            for area, area_metrics_list in metrics_by_area.items():
                if metric in area_metrics_list:
                    slice_colors.append(color_mapping[area])
                    found = True
                    break
            if not found:
                # If the metric is not found in any area, append default colors
                slice_colors.append('#CCCCCC')

        return slice_colors

    slice_colors = get_colors_for_position_group(position_group)
    text_colors = ['#000000'] * 10

    # instantiate PyPizza class
    baker = PyPizza(
        params=metrics,                  # list of parameters
        background_color="#EBEBE9",     # background color
        straight_line_color="#EBEBE9",  # color for straight lines
        straight_line_lw=1,             # linewidth for straight lines
        last_circle_lw=0,               # linewidth of last circle
        other_circle_lw=0,              # linewidth for other circles
        inner_circle_size=20            # size of inner circle
    )

    # plot pizza
    fig, ax = baker.make_pizza(
        values,                          # list of values
        figsize=(8, 8.5),                 # Adjust figsize to make the figure smaller
        color_blank_space="same",        # use same color to fill blank space
        slice_colors=slice_colors,       # color for individual slices
        value_colors=text_colors,        # color for the value-text
        value_bck_colors=slice_colors,   # color for the blank spaces
        blank_alpha=0.4,                 # alpha for blank-space colors
        kwargs_slices=dict(
            edgecolor="#F2F2F2", zorder=2, linewidth=1
        ),                               # values to be used when plotting slices
        kwargs_params=dict(
            color="#000000", fontsize=11,
            fontproperties=font_normal.prop, va="center"
        ),                               # values to be used when adding parameter
        kwargs_values=dict(
            color="#000000", fontsize=11,
            fontproperties=font_normal.prop, zorder=3,
            bbox=dict(
                edgecolor="#000000", facecolor="cornflowerblue",
                boxstyle="round,pad=0.2", lw=1
            )
        )                                # values to be used when adding parameter-values
    )

    # add title
    fig.text(
        0.515, 0.975, f"{player_name} | {position_group} Template | {season_filter} Season", size=16,
        ha="center", fontproperties=font_bold.prop, color="#000000"
    )

    # add subtitle
    fig.text(
        0.515, 0.953,
        f"Percentile Rank vs {extract} {position_group}'s",
        size=13,
        ha="center", fontproperties=font_bold.prop, color="#000000"
    )

    # add credits
    CREDIT_1 = "data: opta viz fbref | using mplsoccer"
    CREDIT_2 = "created by: @ITFCAnalytics inspired by: @Worville, @FootballSlices"

    fig.text(
        0.99, 0.02, f"{CREDIT_1}\n{CREDIT_2}", size=9,
        fontproperties=font_italic.prop, color="#000000",
        ha="right"
    )

    # add text
    fig.text(
        0.15, 0.925, "Defending        Passing       Ball Carrying       Chance Creation       Shooting", size=14,
        fontproperties=font_bold.prop, color="#000000"
    )

    # add rectangles
    fig.patches.extend([
        plt.Rectangle(
            (0.115, 0.9225), 0.025, 0.021, fill=True, color='#1A78CF',
            transform=fig.transFigure, figure=fig
        ),
        plt.Rectangle(
            (0.280, 0.9225), 0.025, 0.021, fill=True, color='#008000',
            transform=fig.transFigure, figure=fig
        ),
        plt.Rectangle(
            (0.416, 0.9225), 0.025, 0.021, fill=True, color='#FFA500',
            transform=fig.transFigure, figure=fig
        ),
        plt.Rectangle(
            (0.612, 0.9225), 0.025, 0.021, fill=True, color='#800080',
            transform=fig.transFigure, figure=fig
        ),
        plt.Rectangle(
            (0.840, 0.9225), 0.025, 0.021, fill=True, color='#FF0000',
            transform=fig.transFigure, figure=fig
        ),
    ])

    st.pyplot(fig)

def find_similar_players(player_name, season_name, position_group, df, n_clusters=20, top_n=5):
    """
    Find similar players using KMeans clustering, limited to players in the same position and season.
    """
    # Get player's data
    player_data = df[(df['Player Name'] == player_name) & (df['Season'] == season_name)]
    if player_data.empty:
        print(f"Player '{player_name}' not found in season '{season_name}'")
        return
    
    #player_position_group = player_data['Position Group'].iloc[0]
    
    # Filter for same position players in the same season
    position_df = df[((df['Position Group'] == position_group) | (df['Player Name'] == player_name)) & (df['Season'] == season_name)]
    
    # Drop duplicates based on Player and Season
    #position_df = position_df.drop_duplicates(subset=['Player Name', 'Season', 'Main Position'], keep='first')
    
    # Define metrics for each position group (same as radar chart)
    metrics_by_position = {
        'FB': [
            'Aerial Ability', 'Defensive Awareness', 'Defensive Intensity', '1v1 Defending', 
            'Pass Progression', 'Pass Retention', 'Ball Carrying',
            'Volume of Take-ons', 'Chance Creation', 'Impact in and around box'
        ],
        'CB': [
            'Aerial Ability', 'Box Defending', 'Defensive Awareness', '1v1 Defending',
            'Defensive Intensity', 'Pass Progression', 'Pass Retention',
            'Switching Play', 'Ball Carrying', 'Shot Volume'
        ],
        'DM': [
            'Aerial Ability', 'Defensive Awareness', 'Defensive Intensity', '1v1 Defending',
            'Pass Progression', 'Pass Retention', 'Switching Play', 'Volume of Take-ons',
            'Retention from Take-ons', 'Shot Volume'
        ],
        'CM': [
            'Defensive Awareness', 'Defensive Intensity', 'Pass Progression', 'Pass Retention',
            'Ball Carrying', 'Volume of Take-ons', 'Retention from Take-ons',
            'Chance Creation', 'Impact in and around box', 'Shot Volume'
        ],
        'AM': [
            'Defensive Intensity', 'Pass Progression', 'Pass Retention',
            'Volume of Take-ons', 'Retention from Take-ons', 'Chance Creation', 'Impact in and around box',
            'Shot Volume', 'Shot Quality', 'Self-created Shots'
        ],
        'W': [
            'Defensive Intensity', 'Pass Retention',
            'Ball Carrying', 'Volume of Take-ons', 'Retention from Take-ons',
            'Chance Creation', 'Impact in and around box',
            'Shot Volume', 'Shot Quality', 'Self-created Shots'
        ],
        'ST': [
            'Aerial Ability', 'Defensive Intensity',
            'Pass Retention', 'Ball Carrying', 'Volume of Take-ons',
            'Chance Creation', 'Impact in and around box',
            'Shot Volume', 'Shot Quality', 'Self-created Shots'
        ]
    }
    
    # Get metrics for player's position
    metrics = metrics_by_position[position_group]
    
    # Create similarity DataFrame with available metrics
    df_similar = position_df[['Player Name', 'Main Position', 'Position Group', 'Squad', 'Season'] + metrics].copy()
    
    # Drop duplicates based on Player
    df_similar = df_similar.drop_duplicates(subset=['Player Name', 'Season', 'Main Position'], keep='first')
    
    # Handle missing values
    df_similar = df_similar.fillna(0)
    
    # Store player names and seasons
    player_names = df_similar['Player Name'].tolist()
    
    # Keep the necessary columns for merging later
    player_info = df_similar[['Player Name', 'Main Position', 'Position Group', 'Squad', 'Season']].copy()
    
    # Drop non-numeric columns for similarity calculation
    df_similar = df_similar.drop(['Player Name', 'Main Position', 'Position Group', 'Squad', 'Season'], axis=1)
    
    # Scale the features using StandardScaler
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(df_similar.values)
    
    # Calculate cosine similarity instead of Euclidean distance
    similarity_matrix = cosine_similarity(x_scaled)
    
    # Create a DataFrame for similarity scores
    sim_df = pd.DataFrame(similarity_matrix, index=player_names, columns=player_names)
    
    # Get similar players
    similar_players = pd.DataFrame({
        'Player Name': sim_df[player_name].index,
        'Similarity %': sim_df[player_name].values * 100
    })
    
    # Sort by similarity and drop duplicates
    similar_players = similar_players.sort_values('Similarity %', ascending=False).drop_duplicates(subset=['Player Name'], keep='first')
    
    # Merge with player information on both Player and Season
    similar_players = similar_players.merge(
        player_info,
        on='Player Name',  # Merge on both Player and Season
        how='left'
    )
    
    # Remove the target player and get top N
    similar_players = similar_players[similar_players['Player Name'] != player_name].head(top_n)
    
    # Convert similarity to integer
    similar_players['Similarity %'] = similar_players['Similarity %'].astype(int)
    
    # Clean up and reorder columns
    similar_players = similar_players[['Player Name', 'Squad', 'Season', 'Main Position', 'Similarity %']]
    
    return similar_players

# Define metrics for each position group
chart_metrics_by_position = {
    'FB': {
        'Pass Types': [
            'ShortPass%_PR',
            'MediumPass%_PR',
            'LongPass%_PR',
            'ProgPass%_PR',
            'Switch%_PR',
            'KeyPass%_PR',
            'Final3rdPass%_PR',
            'ThroughPass%_PR',
            'LineBreakingPass%_PR'
        ],
        'Touch Areas': [
            'TouchesPer90_PR',
            'TouchCentrality_PR',
            'ActionsPerTouch_PR',
            'Def3rdTouch%_PR',
            'Mid3rdTouch%_PR',
            'Att3rdTouch%_PR',
            'AttPenTouch%_PR'
        ],
        'Tackle Areas': [
            'Def3rdTkl%_PR',
            'Mid3rdTkl%_PR',
            'Att3rdTkl%_PR'
        ],
        'Defensive Play': [
            'TklPer90_PR',
            'DrbTkl%_PR',
            'FlsPer90_PR',
            'PassBlocksPer90_PR',
            'IntPer90_PR',
            'LooseBallWinsPer90_PR',
            'RecovPer90_PR',
            'pAdjClrPer90_PR',
            'pAdjShBlocksPer90_PR',
            'AerialWinsPer90_PR',
            'AerialWin%_PR'
        ],
        'Ball Progression and Retention': [
            'PassesCompletedPer90_PR',
            'TotCmp%_PR',
            'Final1/3CmpPer90_PR',
            'ProgPassesPer90_PR',
            'SwitchesPer90_PR',
            'LineBreakingPassesPer90_PR',
            'ReceivedPassPer90_PR',
            'ProgPassesRecPer90_PR'
        ],
        'Ball Carrying and Dribbling': [
            'AttDrbPer90_PR',
            'DrbSucc%_PR',
            'CarriesPer90_PR',
            'CarriesToFinalThirdPer90_PR',
            'CarriesToPenAreaPer90_PR',
            'ProgCarriesPer50Touches_PR',
            'ProgDistancePerCarry_PR',
            'FldPer90_PR'
        ],
        'Creativity and Attacking Play': [
            'AssistsPer90_PR',
            'xAPer90_PR',
            'KeyPassesPer90_PR',
            'PenAreaCmpPer90_PR',
            'ThruBallsPer90_PR',
            'CrsPer90_PR'
        ],
        'Goal Threat': [
            'GoalsPer90_PR',
            'ShotsPer90_PR',
            'npxGPer90_PR',
            'SCAPer90_PR'
        ]
    },
    'CB': {
        'Pass Types': [
            'ShortPass%_PR',
            'MediumPass%_PR',
            'LongPass%_PR',
            'ProgPass%_PR',
            'Switch%_PR',
            'Final3rdPass%_PR',
            'LineBreakingPass%_PR'
        ],
        'Tackle Areas': [
            'Def3rdTkl%_PR',
            'Mid3rdTkl%_PR',
            'Att3rdTkl%_PR'
        ],
        'Defensive Play': [
            'TklPer90_PR',
            'DrbTkl%_PR',
            'FlsPer90_PR',
            'PKconPer90_PR',
            'OGPer90_PR',
            'PassBlocksPer90_PR',
            'IntPer90_PR',
            'LooseBallWinsPer90_PR',
            'RecovPer90_PR',
            'pAdjClrPer90_PR',
            'pAdjShBlocksPer90_PR',
            'AerialWinsPer90_PR',
            'AerialWin%_PR'
        ],
        'Ball Progression and Retention': [
            'PassesCompletedPer90_PR',
            'TotCmp%_PR',
            'Final1/3CmpPer90_PR',
            'ProgPassesPer90_PR',
            'LineBreakingPassesPer90_PR',
            'SwitchesPer90_PR'
        ],
        'Ball Carrying and Dribbling': [
            'CarriesPer90_PR',
            'CarriesToFinalThirdPer90_PR',
            'ProgCarriesPer50Touches_PR',
            'ProgDistancePerCarry_PR'
        ],
        'Goal Threat': [
            'GoalsPer90_PR',
            'ShotsPer90_PR',
            'npxGPer90_PR'
        ]
    },
    'DM': {
        'Pass Types': [
            'ShortPass%_PR',
            'MediumPass%_PR',
            'LongPass%_PR',
            'ProgPass%_PR',
            'Switch%_PR',
            'KeyPass%_PR',
            'Final3rdPass%_PR',
            'ThroughPass%_PR',
            'LineBreakingPass%_PR'
        ],
        'Touch Areas': [
            'TouchesPer90_PR',
            'TouchCentrality_PR',
            'ActionsPerTouch_PR',
            'Def3rdTouch%_PR',
            'Mid3rdTouch%_PR',
            'Att3rdTouch%_PR'
        ],
        'Tackle Areas': [
            'Def3rdTkl%_PR',
            'Mid3rdTkl%_PR',
            'Att3rdTkl%_PR'
        ],
        'Defensive Play': [
            'TklPer90_PR',
            'DrbTkl%_PR',
            'FlsPer90_PR',
            'PassBlocksPer90_PR',
            'IntPer90_PR',
            'LooseBallWinsPer90_PR',
            'RecovPer90_PR',
            'pAdjClrPer90_PR',
            'pAdjShBlocksPer90_PR',
            'AerialWinsPer90_PR',
            'AerialWin%_PR'
        ],
        'Ball Progression and Retention': [
            'PassesCompletedPer90_PR',
            'TotCmp%_PR',
            'Final1/3CmpPer90_PR',
            'ProgPassesPer90_PR',
            'LineBreakingPassesPer90_PR',
            'SwitchesPer90_PR',
            'ReceivedPassPer90_PR',
            'ProgPassesRecPer90_PR'
        ],
        'Ball Carrying and Dribbling': [
            'AttDrbPer90_PR',
            'DrbSucc%_PR',
            'CarriesPer90_PR',
            'CarriesToFinalThirdPer90_PR',
            'ProgCarriesPer50Touches_PR',
            'ProgDistancePerCarry_PR',
            'FldPer90_PR'
        ],
        'Creativity and Attacking Play': [
            'AssistsPer90_PR',
            'xAPer90_PR',
            'KeyPassesPer90_PR',
            'ThruBallsPer90_PR'
        ],
        'Goal Threat': [
            'GoalsPer90_PR',
            'ShotsPer90_PR',
            'npxGPer90_PR'
        ]
    },
    'CM': {
        'Pass Types': [
            'ShortPass%_PR',
            'MediumPass%_PR',
            'LongPass%_PR',
            'ProgPass%_PR',
            'Switch%_PR',
            'KeyPass%_PR',
            'Final3rdPass%_PR',
            'ThroughPass%_PR',
            'LineBreakingPass%_PR'
        ],
        'Touch Areas': [
            'TouchesPer90_PR',
            'TouchCentrality_PR',
            'ActionsPerTouch_PR',
            'Def3rdTouch%_PR',
            'Mid3rdTouch%_PR',
            'Att3rdTouch%_PR',
            'AttPenTouch%_PR'
        ],
        'Tackle Areas': [
            'Def3rdTkl%_PR',
            'Mid3rdTkl%_PR',
            'Att3rdTkl%_PR'
        ],
        'Defensive Play': [
            'TklPer90_PR',
            'DrbTkl%_PR',
            'FlsPer90_PR',
            'PassBlocksPer90_PR',
            'IntPer90_PR',
            'LooseBallWinsPer90_PR',
            'RecovPer90_PR',
            'AerialWinsPer90_PR',
            'AerialWin%_PR'
        ],
        'Ball Progression and Retention': [
            'PassesCompletedPer90_PR',
            'TotCmp%_PR',
            'Final1/3CmpPer90_PR',
            'ProgPassesPer90_PR',
            'LineBreakingPassesPer90_PR',
            'SwitchesPer90_PR',
            'ReceivedPassPer90_PR',
            'ProgPassesRecPer90_PR'
        ],
        'Ball Carrying and Dribbling': [
            'AttDrbPer90_PR',
            'DrbSucc%_PR',
            'CarriesPer90_PR',
            'CarriesToFinalThirdPer90_PR',
            'CarriesToPenAreaPer90_PR',
            'ProgCarriesPer50Touches_PR',
            'ProgDistancePerCarry_PR',
            'FldPer90_PR'
        ],
        'Creativity and Attacking Play': [
            'AssistsPer90_PR',
            'xAPer90_PR',
            'KeyPassesPer90_PR',
            'PenAreaCmpPer90_PR',
            'CrsPenAreaCmpPer90_PR',
            'ThruBallsPer90_PR',
            'CrsPer90_PR'
        ],
        'Goal Threat': [
            'GoalsPer90_PR',
            'ShotsPer90_PR',
            'npxGPer90_PR',
            'npxG/Sh_PR',
            'SCAPer90_PR',
            'SCADribPer90_PR'
        ]
    },
    'AM': {
        'Pass Types': [
            'ShortPass%_PR',
            'MediumPass%_PR',
            'LongPass%_PR',
            'ProgPass%_PR',
            'Switch%_PR',
            'KeyPass%_PR',
            'Final3rdPass%_PR',
            'ThroughPass%_PR',
            'LineBreakingPass%_PR'
        ],
        'Touch Areas': [
            'TouchesPer90_PR',
            'TouchCentrality_PR',
            'ActionsPerTouch_PR',
            'Def3rdTouch%_PR',
            'Mid3rdTouch%_PR',
            'Att3rdTouch%_PR',
            'AttPenTouch%_PR'
        ],
        'Tackle Areas': [
            'Def3rdTkl%_PR',
            'Mid3rdTkl%_PR',
            'Att3rdTkl%_PR'
        ],
        'Defensive Play': [
            'TklPer90_PR',
            'FlsPer90_PR',
            'PassBlocksPer90_PR',
            'IntPer90_PR',
            'LooseBallWinsPer90_PR',
            'AerialWinsPer90_PR',
            'AerialWin%_PR'
        ],
        'Ball Progression and Retention': [
            'PassesCompletedPer90_PR',
            'TotCmp%_PR',
            'Final1/3CmpPer90_PR',
            'ProgPassesPer90_PR',
            'LineBreakingPassesPer90_PR',
            'SwitchesPer90_PR',
            'ReceivedPassPer90_PR',
            'ProgPassesRecPer90_PR'
        ],
        'Ball Carrying and Dribbling': [
            'AttDrbPer90_PR',
            'DrbSucc%_PR',
            'CarriesPer90_PR',
            'CarriesToFinalThirdPer90_PR',
            'CarriesToPenAreaPer90_PR',
            'ProgCarriesPer50Touches_PR',
            'ProgDistancePerCarry_PR',
            'ProgCarryEfficiency_PR',
            'FldPer90_PR'
        ],
        'Creativity and Attacking Play': [
            'AssistsPer90_PR',
            'xAPer90_PR',
            'KeyPassesPer90_PR',
            'PenAreaCmpPer90_PR',
            'CrsPenAreaCmpPer90_PR',
            'ThruBallsPer90_PR',
            'CrsPer90_PR'
        ],
        'Goal Threat': [
            'GoalsPer90_PR',
            'ShotsPer90_PR',
            'SoT%_PR',
            'npxGPer90_PR',
            'npxG/Sh_PR',
            'SCAPer90_PR',
            'SCADribPer90_PR'
        ]
    },
    'W': {
        'Pass Types': [
            'ShortPass%_PR',
            'MediumPass%_PR',
            'LongPass%_PR',
            'ProgPass%_PR',
            'Switch%_PR',
            'KeyPass%_PR',
            'Final3rdPass%_PR',
            'ThroughPass%_PR',
            'LineBreakingPass%_PR'
        ],
        'Touch Areas': [
            'TouchesPer90_PR',
            'TouchCentrality_PR',
            'ActionsPerTouch_PR',
            'Def3rdTouch%_PR',
            'Mid3rdTouch%_PR',
            'Att3rdTouch%_PR',
            'AttPenTouch%_PR'
        ],
        'Tackle Areas': [
            'Def3rdTkl%_PR',
            'Mid3rdTkl%_PR',
            'Att3rdTkl%_PR'
        ],
        'Defensive Play': [
            'TklPer90_PR',
            'FlsPer90_PR',
            'PassBlocksPer90_PR',
            'IntPer90_PR',
            'LooseBallWinsPer90_PR',
            'AerialWinsPer90_PR',
            'AerialWin%_PR'
        ],
        'Ball Progression and Retention': [
            'PassesCompletedPer90_PR',
            'TotCmp%_PR',
            'Final1/3CmpPer90_PR',
            'ProgPassesPer90_PR',
            'LineBreakingPassesPer90_PR',
            'SwitchesPer90_PR',
            'ReceivedPassPer90_PR',
            'ProgPassesRecPer90_PR'
        ],
        'Ball Carrying and Dribbling': [
            'AttDrbPer90_PR',
            'DrbSucc%_PR',
            'CarriesPer90_PR',
            'CarriesToFinalThirdPer90_PR',
            'CarriesToPenAreaPer90_PR',
            'ProgCarriesPer50Touches_PR',
            'ProgDistancePerCarry_PR',
            'ProgCarryEfficiency_PR',
            'FldPer90_PR'
        ],
        'Creativity and Attacking Play': [
            'AssistsPer90_PR',
            'xAPer90_PR',
            'KeyPassesPer90_PR',
            'PenAreaCmpPer90_PR',
            'CrsPenAreaCmpPer90_PR',
            'ThruBallsPer90_PR',
            'CrsPer90_PR'
        ],
        'Goal Threat': [
            'GoalsPer90_PR',
            'ShotsPer90_PR',
            'SoT%_PR',
            'npxGPer90_PR',
            'npxG/Sh_PR',
            'SCAPer90_PR',
            'SCADribPer90_PR'
        ]
    },
    'ST': {
        'Pass Types': [
            'ShortPass%_PR',
            'MediumPass%_PR',
            'LongPass%_PR',
            'ProgPass%_PR',
            'Switch%_PR',
            'KeyPass%_PR',
            'Final3rdPass%_PR',
            'ThroughPass%_PR'
        ],
        'Touch Areas': [
            'TouchesPer90_PR',
            'TouchCentrality_PR',
            'ActionsPerTouch_PR',
            'Def3rdTouch%_PR',
            'Mid3rdTouch%_PR',
            'Att3rdTouch%_PR',
            'AttPenTouch%_PR'
        ],
        'Defensive Play': [
            'TklPer90_PR',
            'FlsPer90_PR',
            'PassBlocksPer90_PR',
            'IntPer90_PR',
            'LooseBallWinsPer90_PR',
            'AerialWinsPer90_PR',
            'AerialWin%_PR'
        ],
        'Ball Progression and Retention': [
            'PassesCompletedPer90_PR',
            'TotCmp%_PR',
            'Final1/3CmpPer90_PR',
            'ProgPassesPer90_PR',
            'SwitchesPer90_PR',
            'ReceivedPassPer90_PR',
            'ProgPassesRecPer90_PR'
        ],
        'Ball Carrying and Dribbling': [
            'AttDrbPer90_PR',
            'DrbSucc%_PR',
            'CarriesPer90_PR',
            'CarriesToFinalThirdPer90_PR',
            'CarriesToPenAreaPer90_PR',
            'ProgCarriesPer50Touches_PR',
            'ProgDistancePerCarry_PR',
            'ProgCarryEfficiency_PR',
            'FldPer90_PR'
        ],
        'Creativity and Attacking Play': [
            'AssistsPer90_PR',
            'xAPer90_PR',
            'KeyPassesPer90_PR',
            'PenAreaCmpPer90_PR',
            'CrsPenAreaCmpPer90_PR',
            'ThruBallsPer90_PR',
            'CrsPer90_PR'
        ],
        'Goal Threat': [
            'GoalsPer90_PR',
            'ShotsPer90_PR',
            'SoT%_PR',
            'npxGPer90_PR',
            'npxG/Sh_PR',
            'AvgShotDistancePer90_PR',
            'SCAPer90_PR',
            'SCADribPer90_PR'
        ]
    }
}
        

def create_player_bars(player_name, season_name, position_group, col, df, chart_type):
    """
    Creates a bar chart for a specified player based on their position group
    and displays it in the specified column.
    """
    # Get player's data and position group
    player_data = df[(df['Player Name'] == player_name) & (df['Season'] == season_name)].iloc[0]
    
    # Get metrics for the specified chart type
    metrics = chart_metrics_by_position[position_group][chart_type]
    values = [round(player_data[metric]) for metric in metrics]
    
    # Normalize values to range from 0 to 100
    normalized_values = np.clip(values, 0, 100)  # Ensure values are within 0-100
    colors = cm.RdYlGn(normalized_values / 100)  # Normalize to [0, 1] for colormap

    with col:  # Use the column context to display the plot
        fig, ax = plt.subplots(figsize=(10, 10))
        bars = ax.barh(metrics, values, color=colors)
        
        # Add dashed grey gridlines
        ax.grid(axis='x', color='white', linestyle='--', linewidth=0.5)
        ax.axvline(x=50, color='white', linewidth=0.8)  # Adjust linewidth as needed
        
        # Remove axes splines
        for s in ['top', 'right']:
            ax.spines[s].set_visible(False)
            ax.spines[s].set_color('white')

        # Change axes color
        for s in ['bottom', 'left']:
            ax.spines[s].set_color('white')

        # Customize the background
        ax.set_facecolor((1, 1, 1, 0))  # Set the background color to transparent
        fig.patch.set_alpha(0)  # Set the figure background to transparent
        
        # Show top values 
        ax.invert_yaxis()
            
        # Remove x, y Ticks
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')

        ax.tick_params(axis='x', colors='white', labelsize=25)  # Change x-axis tick label color
        ax.tick_params(axis='y', colors='white', labelsize=25)  # Change y-axis tick label color

        # Add annotation to bars
        for i in ax.patches:
            plt.text(i.get_width()+0.2, i.get_y()+0.5, 
                     str(round((i.get_width()), 2)),
                     fontsize=20, fontweight='bold',
                     color='white')
            
        # Add Plot Title
        ax.set_title(f"{chart_type} -",
                     loc='left', pad=12.0, fontsize=50, fontweight='normal', color='white', fontname='Arial', style='normal')

        # Set x-axis limits from 0 to 100
        ax.set_xlim(0, 100)

        # Show Plot
        st.pyplot(fig)

def create_player_comparison(player1, player2, season_name_player1, season_name_player2, position_template, df):
    player1_data = df[(df['Player Name'] == player1) & (df['Season'] == season_name_player1)].iloc[0]
    player2_data = df[(df['Player Name'] == player2) & (df['Season'] == season_name_player2)].iloc[0]

    # Get metrics for both players using position template
    metrics = metrics_by_position[position_template]
    values_1 = [round(player1_data[metric]) for metric in metrics]
    values_2 = [round(player2_data[metric]) for metric in metrics]

    # Number of variables
    num_vars = len(metrics)

    # Close the circle for both players
    values_1 += values_1[:1]
    values_2 += values_2[:1]

    # Create angles for each metric
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close the circle for angles

    # Create the radar chart
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Plot both players
    ax.fill(angles, values_1, color='blue', alpha=0.25, label=player1 + ' ' + season_name_player1)  # Fill for Player 1
    ax.plot(angles, values_1, color='blue', linewidth=2)  # Player 1 in blue

    ax.fill(angles, values_2, color='red', alpha=0.25, label=player2 + ' ' + season_name_player2)  # Fill for Player 2
    ax.plot(angles, values_2, color='red', linewidth=2)  # Player 2 in red

    # Set the labels for each angle
    ax.set_xticks(angles[:-1])  # Set the labels for each angle
    ax.set_xticklabels(metrics, fontsize=12, color='white')  # Set the metric names as labels and color them white

    # Rotate the labels to avoid overlap
    for label in ax.get_xticklabels():
        label.set_rotation(0)  # Set rotation to 0 degrees
        label.set_verticalalignment('bottom')  # Align labels to the bottom

    # Remove radial ticks
    ax.set_yticklabels([])  # This removes the radial ticks

    # Customize chart
    plt.title(f"{player1} ({season_name_player1}) vs {player2} ({season_name_player2}) Comparison | {position_template} Template:", pad=20, color='white')

    # Set the background color to transparent
    fig.patch.set_alpha(0)  # Set the figure background to transparent
    ax.set_facecolor((1, 1, 1, 0))  # Set the axes background to transparent

    # Add a legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1))

    # Show the plot in Streamlit
    st.pyplot(fig)

def create_player_traits(player_name, season_name, df):
    player_data = df[(df['Player Name'] == player_name) & (df['Season'] == season_name)]

    traits = [
        'Attempts a lot of dribbles', 'Carries the ball frequently', 'Creates a lot of his own shots',
        'Shoots frequently', 'Attempts a lot of through balls', 'Gets fouled frequently', 
        'Switches the ball frequently', 'Makes a lot of tackles', 'Plays a lot of progressive passes',
        'Plays a lot of short passes', 'Plays a lot of long passes', 'Has a high share of teams total touches',
        'Has a high share of touches in defensive 3rd', 'Has a high share of touches in middle 3rd',
        'Has a high share of touches in final 3rd', 'Has a high share of touches in the penalty box',
        'Competes in a lot of aerial duels', 'Fouls frequently', 'Receives a lot of progressive passes',
        'Has a high share of carries into dangerous areas', 'Crosses the ball frequently', 
        'Shoots from poor areas', 'Shoots from good areas', 'Carries the ball over long distances',
        'Has a high share of tackles in defensive 3rd', 'Has a high share of tackles in middle 3rd',
        'Has a high share of tackles in final 3rd', 'Sweeps up loose balls frequently', 'Blocks passes frequently',
        'Plays a lot of line-breaking passes', 'Takes few touches per action', 'Takes a lot of touches per action'
    ]

    for trait in traits:
        if player_data[trait].iloc[0] == 1:
            st.markdown(
                f'<div style="background-color: #808080; color: white; padding: 1px 1px; border-radius: 5px; display: inline-block; line-height: 1.2;">{trait}</div>', 
                unsafe_allow_html=True
            )

def create_player_strengths_weaknesses(player_name, season_name, df):
    player_data = df[(df['Player Name'] == player_name) & (df['Season'] == season_name)].iloc[0]

    strengths = [
        'Aerial Duels', 'Tackling', 'Passing Completion', 'Long Passing Completion', 'Creating Chances', 'Dribbling', 'Finishing', 'Shooting on Target'
    ]

    weaknesses = [
        'Aerial Duels ', 'Tackling ', 'Passing Completion ', 'Long Passing Completion ', 'Creating Chances ', 'Dribbling ', 'Finishing ', 'Shooting on Target '
    ]

    for strength in strengths:
        if player_data[strength] == 1:
            st.markdown(
                f'<div style="background-color: #358f21; color: white; padding: 1px 1px; border-radius: 5px; display: inline-block; line-height: 1.2;">{strength}</div>', 
                unsafe_allow_html=True
            )

    for weakness in weaknesses:
        if player_data[weakness] == 1:
            st.markdown(
                f'<div style="background-color: #ed3939; color: white; padding: 1px 1px; border-radius: 5px; display: inline-block; line-height: 1.2;">{weakness}</div>', 
                unsafe_allow_html=True
            )

roles_per_position = {
    'CB': [
        'Ball Playing CB', 'Sweeper CB', 'Aggressor', 'Wide CB', 'Box Defender'
    ],
    'FB': [
        'Defensive FB', 'Progressive FB', 'All-Rounder FB', 'Wing Back'
    ],
    'DM': [
        'Deep Lying DM', 'Ball Winning DM', 'Sweeper DM', 'Ball Carrying DM', 'Box-to-Box'
    ],
    'CM': [
        'Deep Lying CM', 'Ball Winning CM', 'Sweeper CM', 'Ball Carrying CM', 'All-Rounder CM', 'Playmaking CM', 'Box Crasher'
    ],
    'AM': [
        'Playmaking 10', 'Second Striker', 'Ball Carrying AM'
    ],
    'W': [
        'Touchline Winger', 'Inside Forward', 'Playmaking Winger', 'Outlet Winger'
    ],
    'ST': [
        'Outlet', 'Target Man', 'Poacher', 'False 9'
    ]
}

def get_color(value):
    """Returns a color based on the value from 1 to 100."""
    # Ensure value is between 1 and 100
    value = max(1, min(100, value))
    
    # Calculate the color
    red = 255 - int((value - 1) * 2.55)  # Red decreases from 255 to 0
    green = int((value - 1) * 2.55)       # Green increases from 0 to 255
    return f'#{red:02x}{green:02x}00'      # Return hex color code

def create_positional_rankings(player_name, season_name, position_group, df):
    """
    Creates positional rankings for a specified player based on their position group
    and displays it in a box.
    """
    # Get player's data and position group
    player_data = df[(df['Player Name'] == player_name) & (df['Season'] == season_name)].iloc[0]

    metrics = roles_per_position[position_group]

    for metric in metrics:
        value = round(player_data[metric])
        color = get_color(value)

        metric_name_html = f'<div style="display: inline-block; padding: 1px 1px; border-radius: 5px; color: white;">{metric}:</div>'
        
        metric_value_html = f'<div style="display: inline-block; padding: 1px 1px; border-radius: 5px; background-color: {color}; color: white;">{value}</div>'
        
        st.markdown(metric_name_html + " " + metric_value_html, unsafe_allow_html=True)

def historic_positional_rankings(player_name, position_group, df):
    """
    Creates a line graph that shows how a player's role has shifted over the years
    """

    plt.clf()

    #df = df.sort_values('Min', ascending=False).drop_duplicates(subset=['Player', 'Season'], keep='first')

    extracted_player_name = player_name.split(' (')[0]

    # Filter the DataFrame for the selected player and season
    player_data = df[(df['Player'] == extracted_player_name)]

    #player_data = player_data.drop_duplicates(subset=['Player Name', 'Season'], keep='first')

    metrics = roles_per_position[position_group]

    # Sort player_data by Season once
    player_data = player_data.sort_values('Season')
    seasons = player_data['Season'].to_numpy()

    for metric in metrics:
        if metric in player_data.columns:
            values = player_data[metric].round().to_numpy()
            plt.plot(seasons, values, label=metric, linewidth=3, marker='o', markersize=6)

    plt.ylim(0, 100)
    plt.xlabel('Season', color='white')
    plt.ylabel('Rating', color='white')
    plt.gca().set_facecolor('none')
    plt.gcf().patch.set_facecolor('none')
    plt.legend(loc='best', framealpha=0.5)

    for spine in plt.gca().spines.values():
        spine.set_color('white')

    plt.tick_params(axis='both', colors='white')

    st.pyplot(plt)

def create_player_information(player_name, season_name, df):
    player_minutes = df[(df['Player Name'] == player_name) & 
                                (df['Season'] == season_name)]['Min'].values[0]
    player_position = df[(df['Player Name'] == player_name) & 
                                (df['Season'] == season_name)]['Main Position'].values[0]
    player_position_group = df[(df['Player Name'] == player_name) & 
                                (df['Season'] == season_name)]['Position Group'].values[0]
    player_squad = df[(df['Player Name'] == player_name) & 
                                (df['Season'] == season_name)]['Squad'].values[0]
    player_age = df[(df['Player Name'] == player_name) & 
                                (df['Season'] == season_name)]['Age'].values[0]
    player_comp = df[(df['Player Name'] == player_name) & 
                                (df['Season'] == season_name)]['Comp'].values[0]
    player_goals = df[(df['Player Name'] == player_name) & 
                                (df['Season'] == season_name)]['Goals'].values[0]
    player_assists = df[(df['Player Name'] == player_name) & 
                                (df['Season'] == season_name)]['Assists'].values[0]

    st.markdown(f"### Player Information:")
    st.markdown(f"Position: {player_position}")
    st.markdown(f"Age: {player_age}")
    st.markdown(f"Competition: {player_comp}")
    st.markdown(f"Squad: {player_squad}")
    st.markdown(f"Minutes Played: {player_minutes}")
    st.markdown(f"Goals: {player_goals}")
    st.markdown(f"Assists: {player_assists}")

def create_filter_dashboard(extract, league, fbref_position, position, season, age_range, club, minutes_range, traits, strengths, weaknesses, df):

    def parse_age(age_str):
        try:
            if isinstance(age_str, (int, float)):  # already numeric
                return age_str
            if '-' in str(age_str):  # e.g. '22-087'
                years, days = age_str.split('-')
                return int(years) + int(days) / 365
            return float(age_str)  # already float in string format
        except:
            return np.nan

    df['Age'] = df['Age'].apply(parse_age)

    #df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    #df['Min'] = pd.to_numeric(df['Min'], errors='coerce')

    df = df[(df['Extract'].isin(extract)) & (df['Comp'].isin(league)) & (df['Pos'].isin(fbref_position)) & (df['Season'].isin(season)) & (df['Age'].between(age_range[0], age_range[1])) & (df['Squad'].isin(club)) & (df['Min'].between(minutes_range[0], minutes_range[1])) & (df['Pos'] != 'GK')]
    #

    position_metrics = roles_per_position[position]
    aggregated_metrics = metrics_by_position[position]

    for trait in traits:
        df = df[df[trait] == 1]

    for strength in strengths:
        df = df[df[strength] == 1]

    for weakness in weaknesses:
        df = df[df[weakness] == 0]

    df = df[['Player', 'Pos', 'Squad', 'Age', 'Comp', 'Season', 'Min'] + position_metrics + aggregated_metrics]

    df = df.sort_values('Min', ascending=False).drop_duplicates(subset=['Player', 'Squad', 'Season', 'Pos'], keep='first')

    # def color_metrics(value):
    #     norm_value = np.clip(value, 0, 100) / 100
    #     color = cm.RdYlGn(norm_value)
    #     return f'background-color: rgba({int(color[0] * 255)}, {int(color[1] * 255)}, {int(color[2] * 255)}, 1);'

    # styled_df = df.style

    # for metric in position_metrics + aggregated_metrics:
    #     styled_df = styled_df.applymap(color_metrics, subset=[metric])

    #styled_df = df.style.applymap(color_percentiles, subset=cols_to_color)

    st.dataframe(df)

## CREATE STREAMLIT DASHBOARD ##

# title, subheader and player/dashboard filter
st.set_page_config(page_title="Player Dashboard -", layout="wide")
st.title(f"Player Dashboard")
st.write("Filter by player and season and then select 'Pizza Chart' for an overview on the player's performance, 'Full Dashboard' for a more detailed look and to see similar players or 'Comparison Dashboard' to compare players performances:")
st.write("Scroll down on the sidebar to view Player Information, Traits, Strengths, Weaknesses and Positional Ranking for your chosen player:") 
st.write("All metrics are percentile ranked within the selected players position group.")

function_filter = st.radio("Select a function to display:", 
                        ("Pizza Chart", "Full Dashboard", "Comparison Dashboard", "Filtering Dashboard"))

if function_filter != "Comparison Dashboard" and function_filter != "Filtering Dashboard":
    unique_players = df_combined['Player Name'].sort_values().unique()
    player_filter = st.selectbox('Select a player:', unique_players, index=0)
    unique_seasons = df_combined[df_combined['Player Name'] == player_filter]['Season'].sort_values().unique()
    season_filter = st.selectbox('Select a season:', unique_seasons, index=0)
    unique_positions = ['FB', 'CB', 'DM', 'CM', 'AM', 'W', 'ST']
    position_filter = st.selectbox('Select a position to compare this player to:', unique_positions, index=0)

    df_combined = create_percentile_rankings(position_filter, player_filter, df_combined)
    create_aggregated_columns(df_combined)

    with st.sidebar:   
        create_player_information(player_filter, season_filter, df_combined)

        st.markdown("---------------")

        st.markdown(f"### Traits:")
        create_player_traits(player_filter, season_filter, df_combined)

        st.markdown("---------------")

        st.markdown(f"### Strengths + Weaknesses:")
        create_player_strengths_weaknesses(player_filter, season_filter, df_combined)

        st.markdown("---------------")

        st.markdown(f"### Positional Ratings:")
        create_positional_rankings(player_filter, season_filter, position_filter, df_combined)

        st.markdown("---------------")

        st.markdown(f"### Positional Ratings Over Time:")
        historic_positional_rankings(player_filter, position_filter, df_combined)


    num_bar_charts = len(chart_metrics_by_position[position_filter])

    # this sets the max columns for a row
    num_columns = 4

    # display each chart for a player
    if player_filter:
        if player_filter in df_combined['Player Name'].values: 
            if function_filter == "Pizza Chart":
                #with st.container():
                st.subheader(f"{player_filter}'s Pizza Chart for {season_filter}:")
                create_player_pizza(player_filter, season_filter, position_filter, df_combined)

            elif function_filter == "Full Dashboard":
                st.subheader(f"{player_filter}'s Bar Chart for {season_filter} | {position_filter} Template:")
                cols = st.columns(num_columns)
                for i in range(num_bar_charts):
                    col_index = i % num_columns
                    chart_type = list(chart_metrics_by_position[position_filter].keys())[i]
                    create_player_bars(player_filter, season_filter, position_filter, cols[col_index], df_combined, chart_type)

                    if col_index == num_columns - 1 and i < num_bar_charts - 1:
                        cols = st.columns(num_columns)

                st.subheader(f"Similar Players to {player_filter}")
                similar_players = find_similar_players(player_filter, season_filter, position_filter, df_combined)
                st.table(similar_players)
        else:
            st.error("Selected player not found in the data.")

elif function_filter == "Comparison Dashboard":
    
    col1, col2 = st.columns(2)

    with col1:
        unique_players1 = df_combined['Player Name'].sort_values().unique()
        player_filter1 = st.selectbox('Select a player:', unique_players1, index=0, key='player1_selectbox')
        unique_seasons1 = df_combined[df_combined['Player Name'] == player_filter1]['Season'].sort_values().unique()
        season_filter1 = st.selectbox('Select a season:', unique_seasons1, index=0, key='season1_selectbox')

    with col2:
        unique_players2 = df_combined['Player Name'].sort_values().unique()
        player_filter2 = st.selectbox('Select a player:', unique_players2, index=0, key='player2_selectbox')
        unique_seasons2 = df_combined[df_combined['Player Name'] == player_filter2]['Season'].sort_values().unique()
        season_filter2 = st.selectbox('Select a season:', unique_seasons2, index=0, key='season2_selectbox')
    
    unique_positions = ['FB', 'CB', 'DM', 'CM', 'AM', 'W', 'ST']
    position_filter = st.selectbox('Select a position to compare this player to:', unique_positions, index=0)

    df_combined = create_percentile_rankings_comparison(position_filter, player_filter1, player_filter2, df_combined)
    create_aggregated_columns(df_combined)

    st.subheader("Player Comparison:")
    create_player_comparison(player_filter1, player_filter2, season_filter1, season_filter2, position_filter, df=df_combined)

    col3, col4 = st.columns(2)

    with col3:
        st.write("---------------")
        st.subheader(f"{player_filter1} ({season_filter1})")
        st.write("---------------")
        create_player_information(player_filter1, season_filter1, df_combined)
        st.write("---------------")

    with col4:
        st.write("---------------")
        st.subheader(f"{player_filter2} ({season_filter2})")
        st.write("---------------")
        create_player_information(player_filter2, season_filter2, df_combined)
        st.write("---------------")

    col5, col6, col7, col8, col9, col10 = st.columns(6)

    with col5:
        st.write("Positional Ratings:")
        create_positional_rankings(player_filter1, season_filter1, position_filter, df_combined)

    with col6:
        st.write("Traits:")
        create_player_traits(player_filter1, season_filter1, df_combined)

    with col7:
        st.write("Strengths + Weaknesses:")
        create_player_strengths_weaknesses(player_filter1, season_filter1, df_combined)

    with col8:
        st.write("Positional Ratings:")
        create_positional_rankings(player_filter2, season_filter2, position_filter, df_combined)

    with col9:
        st.write("Traits:")
        create_player_traits(player_filter2, season_filter2, df_combined)

    with col10:
        st.write("Strengths + Weaknesses:")
        create_player_strengths_weaknesses(player_filter2, season_filter2, df_combined)

    col11, col12 = st.columns(2)

    with col11:
        st.write("---------------")
        st.write("Positional Ratings Over Time:")
        historic_positional_rankings(player_filter1, position_filter, df_combined)

    with col12:
        st.write("---------------")
        st.write("Positional Ratings Over Time:")
        historic_positional_rankings(player_filter2, position_filter, df_combined)

elif function_filter == "Filtering Dashboard":

    with st.sidebar:

        fbref_unique_positions = df_combined['Pos'].sort_values().unique()
        #['DF', 'DF,MF', 'MF', 'MF,FW', 'FW']
        fbref_position_filter = st.multiselect('Select a position:', ['All'] + list(fbref_unique_positions), default='All')
        if 'All' in fbref_position_filter:
            fbref_position_filter = list(fbref_unique_positions)

        unique_positions = ['FB', 'CB', 'DM', 'CM', 'AM', 'W', 'ST']
        position_filter = st.selectbox('Select a position as your template:', unique_positions, index=0)

        df_combined = create_percentile_rankings_filtered(fbref_position_filter, position_filter, df_combined)
        create_aggregated_columns(df_combined)

        unique_extract = ['Top 5 Leagues', 'Next 12 Leagues']
        extract_filter = st.multiselect('Select an extract of leagues:', ['All'] + list(unique_extract), default='All')
        if 'All' in extract_filter:
            extract_filter = list(unique_extract)

        unique_leagues = df_combined[df_combined['Extract'].isin(extract_filter)]['Comp'].sort_values().unique()
        league_filter = st.multiselect('Select a league:', ['All'] + list(unique_leagues), default='All')
        if 'All' in league_filter:
            league_filter = list(unique_leagues)

        unique_seasons = ['2022-2023', '2023-2024', '2024-2025']
        season_filter = st.multiselect('Select a season:', ['All'] + list(unique_seasons), default='All')
        if 'All' in season_filter:
            season_filter = list(unique_seasons)

        age_filter = st.slider('Select an age range:', 16, 45, (16, 45))

        unique_clubs = df_combined[df_combined['Comp'].isin(league_filter)]['Squad'].sort_values().unique()
        club_filter = st.multiselect('Select a club:', ['All'] + list(unique_clubs), default='All')
        if 'All' in club_filter:
            club_filter = list(unique_clubs)

        minutes_filter = st.slider('Select a range of minutes:', 0, 4500, (0, 4500))

    traits = [
        'Attempts a lot of dribbles', 'Carries the ball frequently', 'Creates a lot of his own shots',
        'Shoots frequently', 'Attempts a lot of through balls', 'Gets fouled frequently', 
        'Switches the ball frequently', 'Makes a lot of tackles', 'Plays a lot of progressive passes',
        'Plays a lot of short passes', 'Plays a lot of long passes', 'Has a high share of teams total touches',
        'Has a high share of touches in defensive 3rd', 'Has a high share of touches in middle 3rd',
        'Has a high share of touches in final 3rd', 'Has a high share of touches in the penalty box',
        'Competes in a lot of aerial duels', 'Fouls frequently', 'Receives a lot of progressive passes',
        'Has a high share of carries into dangerous areas', 'Crosses the ball frequently', 
        'Shoots from poor areas', 'Shoots from good areas', 'Carries the ball over long distances',
        'Has a high share of tackles in defensive 3rd', 'Has a high share of tackles in middle 3rd',
        'Has a high share of tackles in final 3rd', 'Sweeps up loose balls frequently', 'Blocks passes frequently',
        'Plays a lot of line-breaking passes', 'Takes few touches per action', 'Takes a lot of touches per action'
    ]

    trait_filter = st.multiselect('Select traits that you want your player to have:', traits)

    strengths = [
        'Aerial Duels', 'Tackling', 'Passing Completion', 'Long Passing Completion', 'Creating Chances', 'Dribbling', 'Finishing', 'Shooting on Target'
    ]

    strength_filter = st.multiselect('Select strengths that you want your player to have:', strengths)

    weaknesses = [
        'Aerial Duels ', 'Tackling ', 'Passing Completion ', 'Long Passing Completion ', 'Creating Chances ', 'Dribbling ', 'Finishing ', 'Shooting on Target '
    ]

    weakness_filter = st.multiselect('Select weaknesses that you do not want your player to have:', weaknesses)

    create_filter_dashboard(extract_filter, league_filter, fbref_position_filter, position_filter, season_filter, age_filter, club_filter, minutes_filter, trait_filter, strength_filter, weakness_filter, df=df_combined)