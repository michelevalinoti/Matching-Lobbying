#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sun Jun 18 15:25:54 2023
@author: michelev
"""

# Importing necessary libraries
import pandas as pd
import numpy as np
import math
import re
import sys, os
import matplotlib.pyplot as plt
from itertools import chain
import statsmodels.api as sm
from patsy import dmatrices
from Base import *  # Importing all from a module named Base
from datetime import datetime
import json
import ast

# Setting up a directory path for data processing
machine = "/Users/michelev/Dropbox/lobbying/"
legi_subfolder = "data/NYS_LegiScan/"

# Function to create DataFrames for legislation data
def createLegislationDataFrames():
    # Define a range of session start years
    session_starts = np.arange(2009, 2025, 2)

    # Initialize dictionaries and DataFrames to store session and bill information
    sessions_dict = {'session_name': [], 'session_id': [], 'session_odd_year': [], 'session_even_year': []}
    bills_tot_df = pd.DataFrame()
    bills_detailed_df = pd.DataFrame()
    people_tot_df = pd.DataFrame()
    sponsors_df = pd.DataFrame()
    history_df = pd.DataFrame()
    missing_ref = {}

    # Loop through each session start year
    for session_start in session_starts:
        print(session_start, flush=True)  # Print the session start year

        # Constructing file paths for CSV and JSON data
        folder_path_csv = os.path.join(machine, legi_subfolder + "csv/" + str(session_start) + "-" + str(session_start+1) + "_General_Assembly/csv/")
        folder_path_json = os.path.join(machine, legi_subfolder + "json/" + str(session_start) + "-" + str(session_start+1) + "_General_Assembly/")
        folder_list_json = os.listdir(folder_path_json + "bill/")

        # Reading bills.csv file and defining columns to keep
        bills_df = pd.read_csv(folder_path_csv + "bills.csv")
        columns_to_keep_bill = ['bill_id', 'session_id', 'bill_number', 'bill_type', 'bill_type_id',
                                'body', 'body_id', 'committee', 'progress', 'referrals', 'sasts', 'sponsors']
        columns_to_keep_sponsors = ['people_id', 'name', 'sponsor_type_id', 'sponsor_order',
                                    'committee_sponsor', 'committee_id']
        
        # Loop through each JSON file in the bill folder
        for json_file in folder_list_json:
            bills_detail = pd.read_json(folder_path_json + "bill/" + json_file, orient='index')
            bills_detail = bills_detail[columns_to_keep_bill]
            bills_detailed_df = pd.concat((bills_detailed_df, bills_detail), axis=0)

            # Processing bill progress information and dates
            progress_bill_df = pd.DataFrame(bills_detail.progress[0])
            if len(progress_bill_df) > 0:
                progress_bill_df['date'] = pd.to_datetime(progress_bill_df.date)
                progress_bill_df['year'] = progress_bill_df.date.dt.year
                progress_bill_df['month'] = progress_bill_df.date.dt.month
                progress_bill_df['day'] = progress_bill_df.date.dt.day
                progress_bill_df['date_number'] = progress_bill_df.date.dt.year + progress_bill_df.date.dt.day_of_year/365
                progress_bill_df['event_within_rank'] = progress_bill_df.groupby('event')['date_number'].rank().astype(int)
                progress_bill_df['event_rank'] = progress_bill_df['date_number'].rank().astype(int)

                # Processing committee and referral information
                committee_bill_df = pd.DataFrame.from_dict(bills_detail.committee[0], orient='index').transpose()
                committee_bill_series = pd.Series(bills_detail.committee[0], dtype='object')
                referrals_bill_df = pd.DataFrame(bills_detail.referrals[0])

                # Additional data processing for referrals and history
                if len(referrals_bill_df)>0:
                    referrals_bill_df['date'] = pd.to_datetime(referrals_bill_df.date)
                    referrals_bill_df_ = referrals_bill_df[['committee_id', 'chamber', 'chamber_id', 'name']]
                    is_contained = referrals_bill_df_[referrals_bill_df_.apply(lambda row: row.equals(committee_bill_series), axis=1)].any().all()
                    referrals_bill_df['event'] = 9
                    history_bill_df = progress_bill_df.merge(referrals_bill_df, on = ['date', 'event'], how = 'left')
                    missing_committees = pd.isna(history_bill_df.committee_id) & (history_bill_df.event == 9)
                    is_missing_committees = sum(missing_committees)
                    if is_missing_committees & is_contained:
                        if min(history_bill_df.loc[missing_committees & (history_bill_df.event == 9),'event_within_rank']) == 1:
                            for col in ['committee_id', 'chamber', 'chamber_id', 'name']:
                                history_bill_df.loc[(history_bill_df.event == 9) & (history_bill_df.event_within_rank == 1), col] = bills_detail.committee[0][col]
                else:
                    history_bill_df = progress_bill_df.copy()

                history_bill_df['bill_id'] = bills_detail.bill_id[0]
                history_bill_df['session_id'] = bills_detail.session_id[0]
                history_df = pd.concat((history_df, history_bill_df), axis=0)

            # Processing sponsor information
            if len(bills_detail.sponsors[0]) > 0:
                sponsors_bill_df = pd.DataFrame(bills_detail.sponsors[0])[columns_to_keep_sponsors]
                sponsors_bill_df['bill_id'] = bills_detail.bill_id[0]
                sponsors_bill_df['session_id'] = bills_detail.session_id[0]
                sponsors_df = pd.concat((sponsors_df, sponsors_bill_df), axis=0)
        
        # Continue processing session data and saving to DataFrames
        session_id = pd.unique(bills_df['session_id'])[0]
        sessions_dict['session_name'].append(str(session_start) + "-" + str(session_start+1))
        sessions_dict['session_odd_year'].append(session_start)
        sessions_dict['session_even_year'].append(session_start+1)
        sessions_dict['session_id'].append(session_id)
        
        people_df = pd.read_csv(folder_path_csv + "people.csv")
        people_df['session_id'] = session_id
        bills_tot_df = pd.concat((bills_tot_df, bills_df), axis=0)
        people_tot_df = pd.concat((people_tot_df, people_df), axis=0)

    # Saving the processed data to CSV files
    bills_detailed_df.to_csv(machine + legi_subfolder + 'bills_details_all.csv')
    history_df.to_csv(machine + legi_subfolder + 'history_all.csv')
    sponsors_df.to_csv(machine + legi_subfolder + 'sponsors_all.csv')
    people_tot_df.set_index('people_id', inplace=True)
    columns_to_upper = ['first_name', 'middle_name', 'last_name', 'suffix', 'nickname']
    people_tot_df[columns_to_upper] = people_tot_df[columns_to_upper].apply(lambda col: col.str.upper())
    people_tot_df[columns_to_upper] = people_tot_df[columns_to_upper].apply(remove_accents_from_string)
    people_tot_df.to_csv(machine + legi_subfolder +'people_all.csv')
    bills_tot_df.set_index('bill_id')
    bills_tot_df.to_csv(machine + legi_subfolder + 'bills_all.csv')
    sessions_tot_df = pd.DataFrame(sessions_dict)
    sessions_tot_df.set_index('session_id', inplace=True)
    sessions_tot_df.to_csv(machine + legi_subfolder + 'sessions_all.csv')

# Function definition for `findCommittees'
def findCommittees():
    
    bills_tot_df = pd.read_csv(machine + legi_subfolder + 'bills_all.csv')
    bills_tot_committees = bills_tot_df.loc[bills_tot_df.committee_id != 0, ['session_id', 'committee_id', 'committee']].drop_duplicates()
    committees = bills_tot_committees.committee.str.split(n=1).str[1]
    chambers =  bills_tot_committees.committee.apply(lambda s: s.split(' ')[0])
    bills_tot_committees['committee_name'] = committees
    bills_tot_committees['chamber_name'] = chambers
    bills_tot_committees[['committee', 'committee_name', 'chamber_name']] = bills_tot_committees[['committee', 'committee_name', 'chamber_name']].apply(lambda col: col.str.upper())
    
    session_starts = np.arange(2019, 2023, 2)
    committee_members_df = pd.DataFrame()
    for session_start in session_starts:
    
        folder_path_json = os.path.join(machine, legi_subfolder + "json/" + str(session_start) + "-" + str(session_start+1) + "_General_Assembly/")
        folder_list_json = os.listdir(folder_path_json + "vote/")
    
        for json_file in folder_list_json:
            print(json_file)
            
            votes_detail = pd.read_json(folder_path_json + "vote/" + json_file, orient = 'index').iloc[0]
            votes = votes_detail['votes']
            committee_desc = votes_detail['desc']
            committee_inferred_name = None
            if 'Committee' in committee_desc:
                parts = committee_desc.split(' Committee')
                # Take the first part
                committee_inferred_name = parts[0]
            committee_members = {'year': session_start,
                                 'committee_description': committee_desc,
                                 'committee_inferred_name': committee_inferred_name,
                                  'chamber_id': votes_detail['chamber_id'],
                                  'people_id_inferred': [vote['people_id'] for vote in votes]}
            committee_members_df = pd.concat((committee_members_df, pd.DataFrame(committee_members)))
            
    committee_members_df = committee_members_df.drop_duplicates()
    committee_members_df = committee_members_df.loc[~pd.isna(committee_members_df.committee_inferred_name),:]
    committee_members_df.committee_inferred_name = committee_members_df.committee_inferred_name.apply(lambda s: s.upper())
    sessions_df =  pd.read_csv(machine + legi_subfolder + 'sessions_all.csv')
    
    def returnOddYear(year):
        
        if year % 2 == 0:
            return year-1
        else:
            return year
    committee_members_df['session_odd_year'] = committee_members_df.year.apply(returnOddYear)
    committee_members_df = committee_members_df.merge(sessions_df[['session_id', 'session_odd_year']], how='left', on = 'session_odd_year')
    committee_members_df.pop('year')
    committee_members_df.pop('committee_description')
    committee_members_df = committee_members_df.drop_duplicates()
    committee_members_df = committee_members_df.merge(bills_tot_committees, how = 'left', left_on = ['session_id', 'committee_inferred_name'], right_on = ['session_id', 'committee'])
    committee_members_df.to_csv(machine + legi_subfolder + 'committees_members_all.csv')
    bills_tot_committees.to_csv(machine + legi_subfolder + 'committees_all.csv')
    
    
#createLegislationDataFrames()
#findCommittees()

#sys.exit()
#%%

ethics_subfolder = "data/NYS_Ethics_Commission/"

def createLobbyingDataFrame(table_name):
    
    print(table_name, flush=True)
    folder_path = os.path.join(machine + ethics_subfolder, table_name)
    # get a list of all files in the folder
    # this might contain folders
    file_list = os.listdir(folder_path)
    
    # create an empty DataFrame to store the concatenated data
    concatenated_data = pd.DataFrame()
    
    # read lobbyist/client list to merge with my ids
    if (table_name != 'lobbyist_list') & (table_name != 'client_list'): 
        lobbyist_list = pd.read_csv(machine + ethics_subfolder + "lobbyist_list" + "_all_sessions.csv")
        lobbyist_list.drop(['lobbyist_list_id', 'individual_lobbyist', 'individual_lobbyist_last_name', 'individual_lobbyist_first_name'], axis=1, inplace=True)
        lobbyist_list.drop_duplicates('principal_lobbyist_id', inplace=True)
        client_list = pd.read_csv(machine + ethics_subfolder + "client_list" + "_all_sessions.csv")
    # iterate over each file/folder and concatenate its data to the DataFrame
    for file in file_list:
        file_path = os.path.join(folder_path, file)
        
        # check that thef file is excel/csv type
        if (file.endswith('.xlsx') | file.endswith('.csv')) & (not file.startswith('~')):
            
            print(file, flush=True)
            df = pd.DataFrame()
            # read and concatenate
            if file.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            elif file.endswith('.csv'):
                skip_rows = 0
                if table_name == 'lobbyist_list':
                    skip_rows = 2
                elif table_name == 'client_list':
                    skip_rows = 5
                df = pd.read_csv(file_path, skiprows = skip_rows)
                
            if table_name == 'lobbyist_list':
                
                df = df.rename(columns = {'PRINCIPAL_LOBBYIST1': 'PRINCIPAL_LOBBYIST',
                                          'Textbox21': 'TOTAL_INDIVIDUAL_LOBBYISTS'})
                
                df[['principal_lobbyist', 'principal_lobbyist_address_line_1', 'principal_lobbyist_address_line_2', 'principal_lobbyist_city_state_phone', 'other_1', 'other_2']] = df['PRINCIPAL_LOBBYIST'] .str.split('\r\n', expand=True)
                df.drop(['other_1', 'other_2'], axis=1, inplace=True)
                # correct systematic mismatch
                missing_col = df['principal_lobbyist_city_state_phone'].isna()
                df.loc[missing_col, 'principal_lobbyist_city_state_phone'] = df.loc[missing_col, 'principal_lobbyist_address_line_2']
                df.loc[missing_col, 'principal_lobbyist_address_line_2'] = None
                
                df[['principal_lobbyist_city', 'principal_lobbyist_state', 'principal_lobbyist_phone']] = df['principal_lobbyist_city_state_phone'].apply(extract_city_state_phone).apply(pd.Series)
                df['principal_lobbyist'] = df['principal_lobbyist'].str.upper()
                df.drop(['PRINCIPAL_LOBBYIST', 'principal_lobbyist_city_state_phone'], axis=1, inplace=True)
                
                df[['individual_lobbyist_last_name', 'individual_lobbyist_first_name', 'other']] = df['INDIVIDUAL_LOBBYIST'].str.split(', ', expand=True)
                merge_col = df['other'].isna() == False
                df.loc[merge_col, 'individual_lobbyist_first_name'] = df.loc[merge_col, 'other'] + ' ' + df.loc[merge_col, 'individual_lobbyist_first_name']
                # Capitalize the first letter of each name and convert the rest to lowercase
                df['individual_lobbyist_first_name'] = df['individual_lobbyist_first_name'].str.capitalize()
                df['individual_lobbyist_last_name'] = df['individual_lobbyist_last_name'].str.capitalize()
                df.drop('other', axis=1, inplace=True)
                
                df['TOTAL_INDIVIDUAL_LOBBYISTS'] = df['TOTAL_INDIVIDUAL_LOBBYISTS'].str.extract(r'(\d+)').astype(int)
                
                
                df['year'] = file.split('_')[0]
                
                
                
            elif table_name == 'client_list':
                
                df = df.rename(columns = {'CCLIENT_ADDRESS': 'CONTRACTUAL_CLIENT',
                                          'BCLIENT_ADDRESS': 'BENEFICIAL_CLIENT'})
                
                for col_name in ['CONTRACTUAL', 'BENEFICIAL']:
                    
                    col_name_new = col_name.lower() + '_client'
                    df[[col_name_new, col_name_new + '_address_line_1', col_name_new + '_address_line_2', col_name_new + '_city', col_name_new + '_phone', 'other_1', 'other_2']]  = df[col_name + '_CLIENT'].str.split('\r\n', expand=True)
                    df[col_name_new] = df[col_name_new].str.upper()
                    df.drop(['other_1', 'other_2'], axis=1, inplace=True)
                    missing_col = df[col_name_new + '_phone'].isna() | (df[col_name_new + '_phone'] == '')
                    df.loc[missing_col, col_name_new + '_phone'] = df.loc[missing_col, col_name_new + '_city']
                    df.loc[missing_col, col_name_new + '_city'] = df.loc[missing_col, col_name_new + '_address_line_2']
                    df.loc[missing_col, col_name_new + '_address_line_2'] = None
                df.drop(['CONTRACTUAL_CLIENT', 'BENEFICIAL_CLIENT'], axis=1, inplace=True)
                
                
                
                df['year'] = int(file.split('_')[0])
                
            elif table_name == 'lobbyist_dataset':
                    
                df.columns = ['principal_lobbyist',
                              'contractual_client',
                              'beneficial_client',
                              'lobbyist_type',
                              'compensation',
                              'reimbursed_expenses',
                              'non_lobbying_expenses',
                              'coalition_contribution']

                for column_name in ['compensation', 'reimbursed_expenses', 'non_lobbying_expenses', 'coalition_contribution']:
                    
                    df[column_name] =  df[column_name].str.replace('$', '').str.replace(',', '').astype(int)


                df['year'] = file.split('_')[0]
                df['retained_flag'] = df['lobbyist_type'] == 'Retained'
                
                for col in ['principal_lobbyist', 'contractual_client', 'beneficial_client']:
                    
                    df[col] = df[col].apply(stripOnlyString)
                    
            elif (table_name == 'state_activities') | (table_name == 'municipal_activities'):
                df = df.rename(columns={df.columns[3]: 'ISSUE'})
                
                pattern = r'\d+_(.*?)\.xlsx'
                match = re.search(pattern, file)
                issue = match.group(1)
                issue = issue.replace('_', ' ')
                issue = issue.replace('&', '/')
                issue = issue.upper()
                df['ACTIVITY'] = issue
                
                df['YEAR'] = df['YEAR'].apply(extract_years)
                df = df.explode('YEAR', ignore_index=True)

                df['YEAR'] = df['YEAR'].astype(int)

                
            
            elif table_name == 'subjects_lobbied':
            
                df['Main Issue'] = df['SUBJECT LOBBIED'].apply(splitSubjectsLobbied)
                df['TOTAL COMPENSATION'] = df['TOTAL COMPENSATION'].apply(extractCompensation)
                df['TOTAL EXPENSES'] = df['TOTAL EXPENSES'].apply(extractCompensation)
                
                df['YEAR'] = df['YEAR'].apply(extract_years)

                df = df.explode('YEAR', ignore_index=True)

                df['YEAR'] = df['YEAR'].astype(int)

            concatenated_data = pd.concat([concatenated_data, df], ignore_index=True)
            
        # do the same if it is a folder
        elif (os.path.isdir(file_path) & (table_name != 'lobbyist_list')):
            
            file_sublist = os.listdir(file_path)
            
            for subfile in file_sublist:
                print(subfile)
                if subfile.endswith('.xlsx') & (not subfile.startswith('~')):
                    
                    df = pd.read_excel(os.path.join(file_path, subfile))
                    if (table_name == 'state_activities') | (table_name == 'municipal_activities'):
                        df = df.rename(columns={df.columns[3]: 'ISSUE'})
                        
                        pattern = r'\d+_(.+)'
                        match = re.search(pattern, file)
                        issue = match.group(1)
                        issue = issue.replace('_', ' ')
                        issue = issue.replace('&', '/')
                        issue = issue.upper()
                        df['ACTIVITY'] = issue
                        
                        df['YEAR'] = df['YEAR'].apply(extract_years)

                        df = df.explode('YEAR', ignore_index=True)

                        df['YEAR'] = df['YEAR'].astype(int)

                        
                    concatenated_data = pd.concat([concatenated_data, df], ignore_index=True)

    concatenated_data.columns = concatenated_data.columns.str.title()
    concatenated_data.columns = concatenated_data.columns.str.replace('_', '')
    concatenated_data.columns = concatenated_data.columns.str.replace(' ', '')
    
    
    
    rename_cols_dict = {key: renameColumn(key) for key in concatenated_data.columns}


    
    # rename the columns
    concatenated_data.rename(rename_cols_dict, axis=1, inplace=True)
    
    if table_name == 'lobbyist_list':
        
        concatenated_data['principal_lobbyist_id'], unique_lobbyists = pd.factorize(concatenated_data['principal_lobbyist'])
        
    elif table_name == 'client_list':
        
        for col in ['contractual_client', 'beneficial_client']:
            concatenated_data[col + '_id'], unique_lobbyists = pd.factorize(concatenated_data[col])

    else:
        concatenated_data = concatenated_data.dropna(subset=['principal_lobbyist', 'contractual_client', 'beneficial_client'])
        concatenated_data.principal_lobbyist = concatenated_data.principal_lobbyist.apply(lambda s: s.split('\n\n')[0].strip())
        concatenated_data.contractual_client = concatenated_data.contractual_client.apply(lambda s: s.split('\n\n')[0].strip())
        concatenated_data.beneficial_client = concatenated_data.beneficial_client.apply(lambda s: s.split('\n\n')[0].strip())
        
        for col in ['principal_lobbyist', 'contractual_client', 'beneficial_client']:
            matched_ids = pd.DataFrame(concatenated_data[col].drop_duplicates())
            if col == 'principal_lobbyist':
                matched_ids[col+'_id'] = matched_ids.apply(fuzzy_match, axis=1, choices=lobbyist_list[col], column_to_match = col)
                matched_ids[col+'_id'] = matched_ids[col+'_id'].apply(lambda idx: lobbyist_list.loc[int(idx),col+'_id'] if pd.isna(idx) == False else None)
            else:
                matched_ids[col+'_id'] = matched_ids.apply(fuzzy_match, axis=1, choices=client_list.drop_duplicates(col+'_id').loc[:,col], column_to_match = col)
                matched_ids[col+'_id'] = matched_ids[col+'_id'].apply(lambda idx: client_list.drop_duplicates(col+'_id').loc[int(idx),col+'_id'] if pd.isna(idx) == False else None)
      
            concatenated_data = concatenated_data.merge(matched_ids, how = 'left', on = col)
            
    if table_name == "parties_lobbied":
        concatenated_data.party_name = concatenated_data.party_name.apply(remove_accents_from_string)
        concatenated_data.insert(concatenated_data.columns.get_loc("govt_body") + 1, "chamber_id", np.zeros(len(concatenated_data)))
        concatenated_data.loc[(concatenated_data.govt_body == "ASSEMBLY") | (concatenated_data.govt_body == "ASSEMBLY COMMITTEE"), "chamber_id"] = 73
        concatenated_data.loc[(concatenated_data.govt_body == "SENATE") | (concatenated_data.govt_body == "SENATE COMMITTEE"), "chamber_id"] = 74

    start = 10**math.floor(math.log10(max(concatenated_data.index)))
    concatenated_data.insert(0, table_name + "_id", range(start, start + len(concatenated_data)))
    concatenated_data.set_index(table_name + "_id", inplace=True)
    
    sessions_df = pd.read_csv(machine + legi_subfolder +'sessions_all.csv')
    
    
    
    concatenated_data.insert(1, 'session_id', concatenated_data.year.apply(lambda year: fromYearToSessionId(year, sessions_df)))
    
    if (table_name != 'lobbyist_list') & (table_name != 'client_list') & (table_name != 'lobbyist_dataset'): 
        concatenated_data.loc[concatenated_data.filing_type.apply(lambda s: 'BI-MONTHLY' in s),'filing_bimonth'] = concatenated_data.loc[concatenated_data.filing_type.apply(lambda s: 'BI-MONTHLY' in s), 'filing_period'].apply(fromFilingPeriodToBimonth)
        concatenated_data.loc[concatenated_data.filing_type.apply(lambda s: 'BI-MONTHLY' in s),'filing_semester'] = concatenated_data.loc[concatenated_data.filing_type.apply(lambda s: 'BI-MONTHLY' in s),'filing_bimonth'].apply(lambda x: (x/3>1)+1)
    concatenated_data.to_csv(machine + ethics_subfolder + table_name + "_all_sessions.csv")
    
    return concatenated_data

query_tables_names = ['lobbyist_list',
                      'client_list',
                      'lobbyist_dataset',
                      'subjects_lobbied',
                      'state_activities',
                      'municipal_activities',
                      'parties_lobbied']

#lobbyist_list = createLobbyingDataFrame(query_tables_names[0])
#client_list = createLobbyingDataFrame(query_tables_names[1])
#lobbyist_dataset = createLobbyingDataFrame(query_tables_names[2])
#subjects_lobbied = createLobbyingDataFrame(query_tables_names[3])
#state_activities = createLobbyingDataFrame(query_tables_names[4])
#municipal_activities = createLobbyingDataFrame(query_tables_names[5])
#parties_lobbied = createLobbyingDataFrame(query_tables_names[6])


expenditures_subfolder = "data/NYS_Expenditures/"


def extractExpenditures():
    
    folder_path = os.path.join(machine + expenditures_subfolder + "Assembly/")
    file_list = os.listdir(folder_path)
    
    
    people_df = pd.read_csv(machine + legi_subfolder +'people_all.csv')
    sessions_df = pd.read_csv(machine + legi_subfolder +'sessions_all.csv')
    
    final_df = pd.DataFrame()
    for file_name in file_list:
        
        if file_name.endswith('.csv'):
        
            print(file_name)
            with open(folder_path + file_name, 'r', encoding='utf-8', errors='ignore') as file:
                df_exp = pd.read_csv(file)
            #df_exp = pd.read_csv(folder_path + file, encoding='ISO-8859-1')
            df_exp.columns = df_exp.columns.str.strip()
            df_exp = df_exp.applymap(lambda x: x.strip() if isinstance(x, str) else x)
            df_exp = df_exp.iloc[df_exp.index > 0, :]
            df_exp.rename({'($)AMOUNT': 'amt'}, inplace=True, axis=1)
            df_exp.amt = df_exp.amt.astype(float)
            df_exp[['start_service_date', 'end_service_date']] = df_exp['SERVICE DATES'].str.extract(r'(\d{2}/\d{2}/\d{2})-(\d{2}/\d{2}/\d{2})')
            
            # Convert the date columns to datetime format
            df_exp['start_service_date'] = pd.to_datetime(df_exp['start_service_date'], format='%m/%d/%y')
            df_exp['end_service_date'] = pd.to_datetime(df_exp['end_service_date'], format='%m/%d/%y')
            
            df_exp['start_year'] = df_exp['start_service_date'].dt.year
            df_exp['end_year'] = df_exp['end_service_date'].dt.year
            
            df_exp['start_month'] = df_exp['start_service_date'].dt.month
            df_exp['end_month'] = df_exp['end_service_date'].dt.month
            
            members_df = df_exp.loc[df_exp['DESCRIPTION'] == 'MEMBER OF ASSEMBLY', ['UNIT', 'start_year']]
            members_df['member_session_id'] = members_df.start_year.apply(lambda year: fromYearToSessionId(year, sessions_df))
    
            # Split the name column into last name and first name
            members_df[['last_name', 'first_name']] = members_df['UNIT'].str.split(', ', expand=True)
            
            members_df.loc[members_df.last_name == 'BICHOTTE', 'last_name'] = 'BICHOTTE HERMELYN'
            members_df.loc[members_df.last_name == 'FORREST', 'last_name'] = 'SOUFFRANT FORREST'
            members_df.loc[members_df.last_name == 'RIVAS-WILLIAMS', 'last_name'] = 'WILLIAMS'
            
             
            def find_best_fuzzy_match(first_name, candidate_first_names):
                best_match = max(candidate_first_names, key=lambda x: fuzz.ratio(first_name, x))
                ratio = fuzz.ratio(first_name, best_match)
                return best_match if ratio >= 70 else None  # Adjust the similarity threshold as needed
            
            # Merge based on the 'last_name' and 'last_name' columns (exact match on last names)
            merged_df = pd.merge(members_df, people_df, left_on=['last_name', 'member_session_id'], right_on = ['last_name', 'session_id'], suffixes = ('_exp', '_legiscan'), how='left')
            homonimous_last_name = merged_df.groupby('last_name').UNIT.count().reset_index()
            homonimous_last_name.columns = ['last_name', 'count_units']
            homonimous_last_name['homonimous_last_name'] = False
            homonimous_last_name.loc[homonimous_last_name.count_units > 1, 'homonimous_last_name'] = True
            
            # Find the best fuzzy match for each first name
            merged_df = merged_df.merge(homonimous_last_name, on = 'last_name', how='left')
            merged_df.loc[merged_df.homonimous_last_name, 'best_fuzzy_match'] = merged_df.loc[merged_df.homonimous_last_name,:].apply(lambda row: find_best_fuzzy_match(row['first_name_exp'], merged_df.loc[merged_df.last_name == row['last_name'], 'first_name_legiscan']), axis=1)
            merged_df= merged_df[((merged_df['homonimous_last_name'] == True) & (merged_df['best_fuzzy_match'] == merged_df['first_name_legiscan'])) | (merged_df['homonimous_last_name'] == False)]
            
            persserv_exp = df_exp.loc[df_exp.UNIT.isin(members_df.UNIT) & (df_exp['EXP TYPE'] == 'PERSSERV'),:]
            persserv_exp = persserv_exp.merge(merged_df[['UNIT', 'people_id', 'session_id']], on = 'UNIT', how = 'left')
            
            def are_strings_similar(str1, str2, threshold=90):
                return fuzz.ratio(str1, str2) >= threshold
            
            # Apply the function to identify similar rows
            not_similar_rows = persserv_exp.apply(lambda row: not are_strings_similar(row['UNIT'], row['PAYEE']), axis=1)
            
            # Filter the DataFrame to exclude similar rows
            persserv_exp = persserv_exp[not_similar_rows]
            
            final_df = pd.concat((final_df, persserv_exp), axis=0)
            
    final_df.to_csv(machine + expenditures_subfolder + 'expenditures_assembly_members.csv')
            
    folder_path = os.path.join(machine + expenditures_subfolder + "Senate/")
    file_list = os.listdir(folder_path)
    
     
    people_df = pd.read_csv(machine + legi_subfolder +'people_all.csv')
    sessions_df = pd.read_csv(machine + legi_subfolder +'sessions_all.csv')
    
    final_df = pd.DataFrame()
    for file_name in file_list:
        
        if file_name.endswith('.csv'):
        
            print(file_name)
            with open(folder_path + file_name, 'r', encoding='utf-8', errors='ignore') as file:
                df_exp = pd.read_csv(file,skiprows=5)
            
            df_exp.columns = df_exp.columns.str.strip()
            df_exp = df_exp.applymap(lambda x: x.strip() if isinstance(x, str) else x)
            df_exp.rename({'AMOUNT': 'amt'}, inplace=True, axis=1)
            df_exp.rename({'TITLE/DESCRIPTION': 'DESCRIPTION'}, inplace=True, axis=1)
            df_exp.amt = df_exp.amt.astype(float)
            df_exp['start_service_date'] = pd.to_datetime(df_exp['REPORT_PERIOD_BEGIN_DATE'], format='%Y-%m-%d', errors='coerce')
            df_exp['end_service_date'] = pd.to_datetime(df_exp['REPORT_PERIOD_END_DATE'], format='%Y-%m-%d', errors='coerce')

            df_exp['start_year'] = df_exp['start_service_date'].dt.year
            df_exp['end_year'] = df_exp['end_service_date'].dt.year
            
            df_exp['start_month'] = df_exp['start_service_date'].dt.month
            df_exp['end_month'] = df_exp['end_service_date'].dt.month
            
            
            df_exp['OFFICE'] = df_exp['OFFICE'].str.replace('SENATOR ', '', regex=False)
            
            members_df = df_exp.loc[df_exp['DESCRIPTION'] == 'MEMBER', ['OFFICE', 'start_year']]
            members_df['member_session_id'] = members_df.start_year.apply(lambda year: fromYearToSessionId(year, sessions_df))
            
            members_df.loc[members_df.OFFICE == 'JOSE G. RIVERA', 'OFFICE'] = 'GUSTAVO RIVERA'
            members_df.loc[members_df.OFFICE == 'FREDERICK J. AKSHAR II', 'OFFICE'] = 'FRED AKSHAR'
            #members_df[members_df.last_name == 'FORREST'] = 'SOUFFRANT FORREST'
            #members_df[members_df.last_name == 'RIVAS-WILLIAMS'] = 'WILLIAMS'
            
             
            def find_best_fuzzy_match(name, candidate_names):
                best_match = max(candidate_names, key=lambda x: fuzz.ratio(name, x))
                ratio = fuzz.ratio(name, best_match)
                return best_match if ratio >= 60 else None  # Adjust the similarity threshold as needed
            
            # Merge based on the 'last_name' and 'last_name' columns (exact match on last names)
            members_df['name'] = members_df.apply(lambda row: find_best_fuzzy_match(row['OFFICE'], people_df.loc[people_df.role_id==2,'name'].drop_duplicates('last').apply(lambda s:s.upper())), axis=1)
            people_df.name = people_df.name.apply(lambda s: s.upper())
            merged_df= members_df.merge(people_df, left_on=['name', 'member_session_id'], right_on = ['name', 'session_id'], how='left')
            
            if members_df.name.isna().sum():
                print("break")
                break
            
            persserv_exp = df_exp.loc[df_exp.OFFICE.isin(members_df.OFFICE) & (df_exp['EXPENSE_TYPE'] == 'PERSONAL SERVICE'),:]
            persserv_exp = persserv_exp.merge(merged_df[['OFFICE', 'people_id', 'session_id']], on = 'OFFICE', how = 'left')
            
            def are_strings_similar(str1, str2, threshold=90):
                return fuzz.ratio(str1, str2) >= threshold
            
            # Apply the function to identify similar rows
            not_similar_rows = persserv_exp.apply(lambda row: not are_strings_similar(row['OFFICE'], row['PAYEE']), axis=1)
            
            # Filter the DataFrame to exclude similar rows
            persserv_exp = persserv_exp[not_similar_rows]
            
            final_df = pd.concat((final_df, persserv_exp), axis=0)
            
    final_df.to_csv(machine + expenditures_subfolder + 'expenditures_senate_members.csv')
        
        
# contributions_subfolder = "data/NYS_Board_Elections/"

# def 

#     folder_path = os.path.join(machine + contributions_subfolder, "ALL_REPORTS_StateCommittee/")
#     contributions_tot_df = pd.read_csv(folder_path + "STATE_COMMITTEE.csv",encoding='latin-1')
#     http://api.followthemoney.org/?dt=1&y=2023&APIKey=636b7371feb3ad6fd7e489d028eed3fa&mode=json

#%%
def cleanPartiesDataFrame(df):
    
    rename_cols_dict = {key: renameColumn(key) for key in df.columns}
    
    # rename the columns
    df.rename(rename_cols_dict, axis=1, inplace=True)
    
    df = df.loc[:, df.columns != 'associated_filings']
    
    
    for col in ['party_name', 'focus', 'principal_lobbyist', 'contractual_client', 'beneficial_client']:
        
        df[col] = df[col].astype(str)
        df[col] = df[col].apply(stripOnlyString)
    
    
    
    #dictionary.remove('cousins')
    
    sessions_df = pd.read_csv(machine + legi_subfolder +'sessions_all.csv')
    committee_df = pd.read_csv(machine + legi_subfolder + 'committees_all.csv', index_col = 0)
    
    people_df = pd.read_csv(machine + legi_subfolder +'people_all.csv')
    
    # Load spaCy's English model
    
    #nlp = spacy.load("en_core_web_lg")
    dictionary = set(nltk_words.words())#enchant.Dict("en_US")
    
    def lowerAndSplit(string):
        
        if pd.isna(string) == False:
            
            return str(string).lower().split(' ')
        else:
            
            return 'the'
        
    dictionary.difference_update(set(chain.from_iterable(people_df.first_name.apply(lowerAndSplit).values)))
    dictionary.difference_update(set(chain.from_iterable(people_df.last_name.apply(lowerAndSplit).values)))
    dictionary.difference_update(set(chain.from_iterable(people_df.nickname.apply(lowerAndSplit).values)))
    dictionary.add('assemblymember')
    dictionary.add('assemblywoman')
    dictionary.add('am')
    
    drop_words = ['NYS', 'PUERTO', 'RICAN']
    
    new_nicknames = {'ALFRED': 'AL',
                     'CHRISTOPHER': 'CHRIS',
                     'ELIZABETH': 'LIZ',#['BETTY', 'LIZ'],
                     'JACOB': 'JAKE',
                     'JONATHAN': 'JON',
                     'JOSEPH': 'JOE',
                     'ROBERT': 'BOB',
                     'WILLIAM': 'WILL'}
    
    new_nicknames_temp = new_nicknames.copy()
    for key in new_nicknames_temp.keys():
        
        new_nicknames[new_nicknames_temp[key]] = key
        
    dictionary.difference_update([key.lower() for key in new_nicknames.keys()])
    
    substitute_manual = {'TAYLOR RAYNOR': [21379, 'MEMBER'], # changed last name (maiden name?) - a.k.a. found on Wikipedia https://en.wikipedia.org/wiki/Taylor_Darling_(politician)
                         'JORDINE JONES': [13939, 'STAFF'], # staff member - from LinkedIn https://www.linkedin.com/in/jordine-jones-7378848/
                         'CAROLINE BURKE': [1424, 'STAFF'], # staff member - from LinkedIn https://www.linkedin.com/in/carolyn-burke-999161181/
                         'VICTORIA CLARK': [1402, 'STAFF'], # staff member - from LinkedIn https://www.linkedin.com/in/victoria-clark-8035ba6/
                         'ARIANA KAPLAN': [18602, 'STAFF'], # staff member - from LinkedIn https://www.linkedin.com/in/arianamclark/
                         'EILEEN MILLER': [None, 'STAFF'] # director of communications - https://www.linkedin.com/in/eileen-miller-26695b4/
                         }
    
    def compute_distance(array1, array2):
        
        return np.vectorize(jarowinkler_similarity)(array1[:, np.newaxis], array2)
    
    people_df_dict = {}
    committee_df_dict = {}
    homonim_names_dict = {}
    similar_names_dict = {}
    
    for session_id in sessions_df.session_id.values:
        
        people_session_df = people_df.loc[(people_df.session_id == session_id) & (people_df.committee_id ==0), ['people_id', 'session_id', 'role_id', 'name', 'first_name', 'middle_name', 'last_name', 'suffix', 'nickname']]
        people_df_dict[session_id] = people_session_df
       
        sims = compute_distance(people_session_df.last_name, people_session_df.last_name)
        sims_df = pd.DataFrame(sims)
        sims_df.index = people_session_df.last_name;
        sims_df.columns = people_session_df.last_name;
        
        homonims_dict = {}
        similar_dict = {}
        
        for i in range(np.shape(sims)[1]):
            i_id = people_session_df.iloc[i].people_id
            for j in range(np.shape(sims)[1]):
                if (sims[i,j] >= 0.95) & (sims[i,j] < 1):
                    if i_id in similar_dict.keys():
                        similar_dict[i_id].append(people_session_df.iloc[j].people_id)
                    else:
                        similar_dict[i_id] = [people_session_df.iloc[j].people_id]
                        
        
        for i in range(np.shape(sims)[1]):
            i_id = people_session_df.iloc[i].people_id
            for j in range(np.shape(sims)[1]):
                if (i!=j) & (sims[i,j] == 1):
                    if i_id in similar_dict.keys():
                        homonims_dict[i_id].append(people_session_df.iloc[j].people_id)
                    else:
                        homonims_dict[i_id] = [people_session_df.iloc[j].people_id]
        
        people_df_dict[session_id].set_index('people_id', inplace=True)
        
        similar_names_dict[session_id] = people_session_df.loc[similar_dict.keys()]
        homonim_names_dict[session_id]  = people_session_df.loc[homonims_dict.keys()]
        
        committee_df_dict[session_id] = committee_df.loc[committee_df.session_id == session_id,:]
        
        
    
    
    
    
    
    unique_names_df = df.loc[df[['party_name', 'chamber_id']].drop_duplicates().index, ['party_name', 'chamber_id', 'parties_lobbied_id']];
    
    
    
    human_names = unique_names_df['parties_lobbied_id'].apply(lambda identifier: manipulatePartyName(identifier, df, dictionary, drop_words, substitute_manual, new_nicknames, committee_df_dict, people_df_dict, homonim_names_dict, similar_names_dict))
    
    human_names_dict = []
    
    for idx, value in human_names.items():
        if isinstance(value, dict):
            human_names_dict.append({'parties_lobbied_id': unique_names_df.loc[idx, 'parties_lobbied_id'], **value})
        elif isinstance(value, list):
            for sub_dict in value:
                human_names_dict.append({'parties_lobbied_id': unique_names_df.loc[idx, 'parties_lobbied_id'], **sub_dict})
    
    human_names_dict = pd.DataFrame(human_names_dict)
    human_names_dict = human_names_dict.merge(unique_names_df)
    
    df = df.merge(human_names_dict, on = ['party_name', 'chamber_id'], how = 'left', suffixes=('', '_cleaning'))
    
    def countPieces(string):
        
        return len(string.split(' '))
    
    df['count_pieces'] = df.party_name.apply(countPieces)
    
    df['bill_number'] = df.focus.apply(findBillCode)
    df['is_matched_bill'] = (df['bill_number'].apply(len) == 6) & ((df['bill_number'].apply(lambda s: s[0]) == 'A') | (df['bill_number'].apply(lambda s: s[0]) == 'S'))
    
    
    return df
        
    # Remove leading and trailing newlines from all columns
#%%


#%%
parties_lobbied = pd.read_csv(machine + ethics_subfolder + 'parties_lobbied_all_sessions.csv')
lobbyist_list = pd.read_csv(machine + ethics_subfolder + 'lobbyist_list_all_sessions.csv')

lobbyist_dataset = pd.read_csv(machine + ethics_subfolder + 'lobbyist_dataset_all_sessions.csv')
subjects_lobbied = pd.read_csv(machine + ethics_subfolder + 'subjects_lobbied_all_sessions.csv')
state_activities = pd.read_csv(machine + ethics_subfolder + 'state_activities_all_sessions.csv')
municipal_activities = pd.read_csv(machine + ethics_subfolder + 'municipal_activities_all_sessions.csv')
state_activities.rename({'activity': 'state_activity', 'issue': 'focus'}, axis=1, inplace=True)
municipal_activities.rename({'activity': 'municipal_activity', 'issue': 'focus'}, axis=1, inplace=True)
state_activities.loc[state_activities.state_activity == 'LEGISLATIVE BILLS','state_activity'] = 'LEGISLATIVE BILL'
state_activities.loc[state_activities.state_activity == 'LEGISLATIVE RESOLUTIONS','state_activity'] = 'LEGISLATIVE RESOLUTION'
municipal_activities.loc[municipal_activities.municipal_activity == 'MUNICIPAL BILLS','municipal_activity'] = 'MUNICIPAL BILL'
lobbyist_dataset.rename({'compensation': 'annual_compensation'}, axis=1, inplace=True)
lobbyist_dataset.rename({'reimbursed_expenses': 'annual_reimbursed_expenses'}, axis=1, inplace=True)
lobbyist_dataset.rename({'non_lobbying_expenses': 'annual_non_lobbying_expenses'}, axis=1, inplace=True)
lobbyist_dataset.rename({'coalition_contribution': 'annual_coalition_contribution'}, axis=1, inplace=True)

parties_lobbied = parties_lobbied.merge(lobbyist_list.loc[:,['principal_lobbyist_id', 'session_id', 'total_individual_lobbyists']].drop_duplicates(), on = ['principal_lobbyist_id', 'session_id'])
parties_lobbied = parties_lobbied.merge(lobbyist_dataset.loc[:,['year', 'principal_lobbyist_id', 'beneficial_client_id', 'retained_flag', 'annual_compensation', 'annual_reimbursed_expenses', 'annual_non_lobbying_expenses', 'annual_coalition_contribution']], on = ['year', 'principal_lobbyist_id', 'beneficial_client_id'], how = 'left')
parties_lobbied = parties_lobbied.merge(subjects_lobbied.loc[:,['filing_type', 'year', 'filing_period', 'principal_lobbyist', 'contractual_client', 'beneficial_client', 'main_issue', 'total_compensation']], on = ['filing_type', 'year', 'filing_period', 'principal_lobbyist', 'contractual_client', 'beneficial_client'], how = 'left')
group_on_activities = ['filing_type', 'year', 'filing_period', 'communication', 'principal_lobbyist', 'contractual_client', 'beneficial_client', 'focus']
parties_lobbied = parties_lobbied.merge(state_activities.drop_duplicates(group_on_activities).loc[:,group_on_activities + ['state_activity']], on = group_on_activities, how = 'left')
parties_lobbied = parties_lobbied.merge(municipal_activities.drop_duplicates(group_on_activities).loc[:,group_on_activities + ['municipal_activity']], on = group_on_activities, how = 'left')


def select_value(row):
    if not pd.isna(row['focus_level']):
        return row['focus_level']
    elif not pd.isna(row['state_activity']):
        return row['state_activity']
    else:
        return row['municipal_activity']
    
parties_lobbied['activity'] = parties_lobbied.apply(select_value, axis=1)




parties_lobbied.to_csv(machine + ethics_subfolder + 'parties_lobbied_merged.csv')

cond_gov = ((parties_lobbied.govt_body == 'ASSEMBLY') | (parties_lobbied.govt_body == 'ASSEMBLY COMMITTEE'))
cond_gov = cond_gov | ((parties_lobbied.govt_body == 'SENATE') | (parties_lobbied.govt_body == 'SENATE COMMITTEE'))
cond = cond_gov & ((parties_lobbied.communication == 'DIRECT LOBBYING') | (parties_lobbied.communication == 'BOTH DIRECT AND GRASSROOTS'))
cond = cond & ((parties_lobbied.focus_level == 'LEGISLATIVE BILL'))
cond = cond & ((parties_lobbied.filing_type=='BI-MONTHLY') | (parties_lobbied.filing_type=='BI-MONTHLY AMENDMENT'))

parties_lobbied[cond].to_csv(machine + ethics_subfolder + 'parties_lobbied_legislation.csv')

#%%
seven = pd.read_csv(machine + ethics_subfolder + 'Registered_Lobbyist_Disclosures__7__Year_Window.csv')

sgroup = seven.groupby(['Reporting Year', 'Lobbyist Type', 'Lobbyist Name', 'Client Name'])
#%%

parties_df = pd.read_csv(machine + ethics_subfolder + 'parties_lobbied_legislation.csv')
cleaned = cleanPartiesDataFrame(parties_df)
cleaned.to_csv(machine + ethics_subfolder + 'parties_lobbied_legislation_cleaned.csv')

sys.exit()

#%%

# Define columns to drop
columns_to_drop = ['check_rule', 'full_name', 'full_name_with_nickname', 'full_name_wo_middle_name', 'full_name_with_alternative_nickname', 'possible_error', 'count_pieces', 'committee', 'suffix', 'first_name', 'last_name', 'middle_name', 'suffix', 'nickname']

parties_df = pd.read_csv(machine + ethics_subfolder + 'parties_lobbied_legislation_cleaned.csv', index_col=0)
parties_df['beneficial_client'] = parties_df['beneficial_client'].str.split('\n')
parties_df = parties_df.explode('beneficial_client', ignore_index=True)
parties_df['chamber_id'] = parties_df['chamber_id'].astype(int)
parties_df['filing_bimonth'] = parties_df['filing_bimonth'].astype(int)
parties_df['filing_semester'] = parties_df['filing_semester'].astype(int)
parties_df['contacted_staff_counsel'] = parties_df['contacted_staff_counsel'].astype(bool)
parties_df['is_staff_counsel'] = parties_df['is_staff_counsel'].astype(bool)
parties_df['entire_body'] = parties_df['entire_body'].astype(bool)
parties_df['majority_body'] = parties_df['majority_body'].astype(bool)
#parties_df['people_id'] = parties_df['people_id'].astype(int)
#parties_df['committee_id'] = parties_df['committee_id'].astype(int)
parties_df['is_matched_bill'] = parties_df['is_matched_bill'].astype(bool)

def days_until_month_start(month, year):
    try:
        month = int(month)
        year = int(year)
        if 1 <= month <= 12:
            first_day = pd.Timestamp(f"{year}-{month}-01")
            start_of_year = pd.Timestamp(f"{year}-01-01")
            days_until_start_of_month = (first_day - start_of_year).days
            return days_until_start_of_month - 1
        else:
            raise ValueError("Month should be an integer between 1 and 12.")
    except ValueError as e:
        print(e)
        return None
    
def days_until_month_end(month, year):
    try:
        month = int(month)
        year = int(year)
        if 1 <= month <= 12:
            flast_day = pd.Timestamp(f"{year}-{month}-{calendar.monthrange(year, month)[1]}")
            start_of_year = pd.Timestamp(f"{year}-01-01")
            days_until_end_of_month = (last_day - start_of_year).days
            return days_until_start_of_month
        else:
            raise ValueError("Month should be an integer between 1 and 12.")
    except ValueError as e:
        print(e)
        return None
    
#parties_df['filing_start']
parties_df = parties_df.drop(columns=columns_to_drop)

# Define columns to keep for bills_df
# columns_to_keep = ['bill_id', 'session_id', 'bill_number', 'bill_type', 'bill_type_id', 'body', 'body_id']

# Read bills_df
bills_df = pd.read_csv(machine + legi_subfolder + 'bills_details_all.csv')
# bills_df = bills_df[columns_to_keep]
# Restrict to bill (exclude resolutions)
bills_df = bills_df[bills_df['bill_type_id'] == 1]
bills_df = bills_df[bills_df['session_id'].isin([1644, 1813])]

# Read history_df
history_df = pd.read_csv(machine + legi_subfolder + 'history_all.csv')
history_df = history_df[history_df['session_id'].isin([1644, 1813])]
history_df.rename(columns={'name': 'committee_name'}, inplace=True)
history_df['date'] = pd.to_datetime(history_df['date'])
history_df['year'] = history_df['date'].dt.year
history_df['month'] = history_df['date'].dt.month
history_df['day'] = history_df['date'].dt.day
history_df['date_number'] = history_df['year'] + history_df['date'].dt.dayofyear / 365
history_df['event_rank'] = history_df.groupby('bill_id')['date_number'].rank(method='first').astype(int)
history_df['filing_bimonth'] = (history_df['date'].apply(lambda x: x.month) + 1)//2


#%%

def extract_sast_bill_id(json_str):
    if pd.notna(json_str):
        json_str = json_str.replace("'", '"')
        data = json.loads(json_str)
        return [d['sast_bill_id'] for d in data]
    else:
        return None
    
    
def extract_committee(json_str):
    if (pd.notna(json_str)) & (len(json_str) > 2):
        json_str = json_str.replace("'", '"')
        json_str = json_str.replace("Veterans\"", "Veterans'")
        json_str = json_str.replace("Women\"s", "Women's")
        data = json.loads(json_str)
        return data['committee_id']
    else:
        return None
    
def extract_passed(json_str):
    passed = False
    if (pd.notna(json_str)) & (len(json_str) > 2):
        json_str = json_str.replace("'", '"')
        data = json.loads(json_str)
        events = [d['event'] for d in data]
        if 8 in events:
            passed=True
            
    return passed
    

# Apply the function to create a new column
bills_df['sast_bill_id'] = bills_df['sasts'].apply(extract_sast_bill_id)

# Create new columns based on the 'sast_bill_id' list
bills_df['sast_bill_id_1'] = bills_df['sast_bill_id'].apply(lambda x: x[0] if x else None)
bills_df['sast_bill_id_2'] = bills_df['sast_bill_id'].apply(lambda x: x[1] if len(x) > 1 else None)
bills_df['sast_bill_id_3'] = bills_df['sast_bill_id'].apply(lambda x: x[2] if len(x) > 2 else None)
bills_df['sast_bill_id_4'] = bills_df['sast_bill_id'].apply(lambda x: x[3] if len(x) > 3 else None)

bills_df['committee_id'] = bills_df['committee'].apply(extract_committee)

bills_df['passed'] = bills_df['progress'].apply(extract_passed)
def sort_and_combine(row):
    sorted_values = sorted([int(val) for val in row if not pd.isna(val)])
    combined_value = int(''.join(map(str, sorted_values))) if sorted_values else None
    return combined_value

# Apply the function to create the 'SortedSum' column
bills_df['bill_id_new'] = bills_df[['bill_id', 'sast_bill_id_1', 'sast_bill_id_2', 'sast_bill_id_3', 'sast_bill_id_4']].apply(sort_and_combine, axis=1)

people_df =  pd.read_csv(machine + legi_subfolder + 'people_all.csv')

committees_df = pd.read_csv(machine + legi_subfolder + 'committees_all.csv')

#%%

#history_df = history_df.drop_duplicates(subset = ['event', 'date', 'bill_id'], keep = 'last')

#%%
event_table = history_df.merge(bills_df, on = ['session_id', 'bill_id'], suffixes=('_event', '_bill'), how = 'inner')
event_table = event_table.drop_duplicates(subset = ['event', 'date', 'bill_id_new'], keep = 'last')


event_grouped = event_table.groupby('bill_id_new')

ct = 0
remove_indices = []
for name, group in event_grouped:
    
    no_intros = sum(group.event == 1)
    if no_intros > 1:
        intro_df = group.loc[group.event == 1]
        for i in range(no_intros):
            date_intro = intro_df.iloc[i]['date']
            bill_intro = intro_df.iloc[i]['bill_id']
            if len(group.loc[(group.date == date_intro) & (group.bill_id != bill_intro) & (group.event == 9),:]) > 0:
                if date_intro >= group.loc[(group.date == date_intro) & (group.bill_id != bill_intro) & (group.event == 9),'date'].min():
                    remove_indices.append(intro_df.iloc[i].name)
                    print(name)
                    ct +=1
                    
event_table = event_table.drop(remove_indices)


event_grouped = event_table.groupby('bill_id_new')

ct = 0
remove_indices = []
for name, group in event_grouped:
    
    no_intros = sum(group.event == 1)
    if no_intros > 1:
        
        event_table.loc[event_table.bill_id_new == name, 'bill_id_new'] = event_table.loc[event_table.bill_id_new == name, 'bill_id'].iloc[0] 
    

def extract_info(bill_id_new):
    
    print(bill_id_new)
    event_table_id = event_table.loc[event_table.bill_id_new == bill_id_new, :]
    referrals = event_table_id.referrals
    sponsors = event_table_id.sponsors
    referrals_list_temp = referrals.apply(ast.literal_eval).values.sum()
    
    referrals_set = set()
    
    # Initialize a result list
    referrals_list = []
    
    for d in referrals_list_temp:
        # Convert each dictionary to a frozenset to make it hashable and check for uniqueness
        dict_as_set = frozenset(d.items())
        
        # Check if it's unique and add it to the result list and set
        if dict_as_set not in referrals_set:
            referrals_set.add(dict_as_set)
            referrals_list.append(d)


    committee_chamber_1 = []
    committee_chamber_2 = []
    
    chambers = event_table_id.body_id
    chambers_unique = list(dict.fromkeys(chambers))
    
    chamber_1 = None
    chamber_2 = None
    committee_chamber_1_type_1 = None
    committee_chamber_1_type_2 = []
    committee_chamber_2_type_1 = None
    committee_chamber_2_type_2 = []
    sponsors_chamber_1_type_1 = []
    sponsors_chamber_1_type_2 = []
    sponsors_chamber_2_type_1 = []
    sponsors_chamber_2_type_2 = []
    
    chamber_1 = int(chambers_unique[0])
    if len(chambers_unique) > 1:
        chamber_2 = int(chambers_unique[1])
      
    for i in range(len(referrals_list)):
        referrals_i = referrals_list[i]
        if referrals_i['chamber_id'] == chamber_1:
            committee_chamber_1.append(referrals_i['committee_id'])
        else:
            committee_chamber_2.append(referrals_i['committee_id'])
    
    committee_chamber_1 = list(dict.fromkeys(committee_chamber_1))
    committee_chamber_2 = list(dict.fromkeys(committee_chamber_2))
    
    print(committee_chamber_1)
    committee_chamber_1_type_1 = committee_chamber_1[0]
    if len(committee_chamber_1)>1:
        committee_chamber_1_type_2 = committee_chamber_1_type_2 + committee_chamber_1[1:]
    if len(committee_chamber_2)>0:
        committee_chamber_2_type_1 = committee_chamber_2[0]
        if len(committee_chamber_2)>1:
            committee_chamber_2_type_2 = committee_chamber_2_type_2 + committee_chamber_2[1:]
    
    
    sponsors_chamber_1 = event_table_id.loc[event_table_id.body_id == chamber_1, 'sponsors']
    sponsors_chamber_2 = event_table_id.loc[event_table_id.body_id == chamber_2, 'sponsors']
    
    if len(sponsors_chamber_1) > 0:
        
        sponsors_chamber_1 = sponsors_chamber_1.iloc[0]
        sponsors_chamber_1 = ast.literal_eval(sponsors_chamber_1)
        
        for j in range(len(sponsors_chamber_1)):
            
            sponsors_chamber_1_j = sponsors_chamber_1[j]
            if sponsors_chamber_1_j['sponsor_type_id'] == 1:
                sponsors_chamber_1_type_1.append((sponsors_chamber_1_j['people_id'], int(sponsors_chamber_1_j['party_id'])))
            else:
                sponsors_chamber_1_type_2.append((sponsors_chamber_1_j['people_id'], int(sponsors_chamber_1_j['party_id'])))
            
    if len(sponsors_chamber_2) > 0:
        
        sponsors_chamber_2 = sponsors_chamber_2.iloc[0]
        sponsors_chamber_2 = ast.literal_eval(sponsors_chamber_2)
        
        for j in range(len(sponsors_chamber_2)):
            sponsors_chamber_2_j = sponsors_chamber_2[j]
            if sponsors_chamber_2_j['sponsor_type_id'] == 1:
                sponsors_chamber_2_type_1.append((sponsors_chamber_2_j['people_id'], int(sponsors_chamber_2_j['party_id'])))
            else:
                sponsors_chamber_2_type_2.append((sponsors_chamber_2_j['people_id'], int(sponsors_chamber_2_j['party_id'])))
        
    
    dictionary = {'bill_id_new': bill_id_new,
                  'chamber_1': chamber_1,
                  'chamber_2': chamber_2,
                  'sponsors_chamber_1_type_1': sponsors_chamber_1_type_1,
                  'sponsors_chamber_1_type_2': sponsors_chamber_1_type_2,
                  'sponsors_chamber_2_type_1': sponsors_chamber_2_type_1,
                  'sponsors_chamber_2_type_2': sponsors_chamber_2_type_2,
                  'committee_chamber_1_type_1': committee_chamber_1_type_1,
                  'committee_chamber_1_type_2': committee_chamber_1_type_2,
                  'committee_chamber_2_type_1': committee_chamber_2_type_1,
                  'committee_chamber_2_type_2': committee_chamber_2_type_2
                  }
        
    return dictionary  
    
    #return chamber_1, chamber_2, sponsors_chamber_1_type_1, sponsors_chamber_1_type_2, sponsors_chamber_2_type_1, sponsors_chamber_2_type_2, committee_chamber_1_type_1, committee_chamber_1_type_2, committee_chamber_2_type_1, committee_chamber_2_type_2

dicts = []

for bill_id_new in set(event_table['bill_id_new']):
    
    dicts.append(extract_info(bill_id_new))

event_info_temp = pd.DataFrame(dicts)
    

committee_members_df = pd.read_csv(machine + legi_subfolder + 'committees_members_all_manual.csv')

event_table = event_table.merge(event_info_temp, how = 'left', on = 'bill_id_new')

event_table = event_table.merge(committee_members_df.loc[~pd.isna(committee_members_df.chair),['session_id', 'people_id_inferred', 'committee_id']], left_on = ['session_id', 'committee_chamber_1_type_1'], right_on = ['session_id', 'committee_id'], how='left')
event_table.rename({'people_id_inferred': 'chair_committee_chamber_1_type_1'}, axis=1,inplace=True)
event_table.pop('committee_id')

event_table.to_csv(machine + 'tables/' + 'event_table.csv')

new_bill_id_df = event_table[['bill_id', 'bill_id_new']].drop_duplicates()

#bills_df.pop('bill_id_new')

bills_df = bills_df.merge(new_bill_id_df, on = 'bill_id', suffixes = ('_bill', '_event'), how = 'left')

for missing_bill in bills_df.loc[bills_df.bill_id_new_event.isna(), 'bill_id_new_bill']:
    
    if len(bills_df.loc[(bills_df.bill_id_new_bill == missing_bill) & (~bills_df.bill_id_new_event.isna()), 'bill_id_new_event']) > 0:
        bills_df.loc[(bills_df.bill_id_new_bill == missing_bill) & bills_df.bill_id_new_event.isna(), 'bill_id_new_event'] = bills_df.loc[(bills_df.bill_id_new_bill == missing_bill) & (~bills_df.bill_id_new_event.isna()), 'bill_id_new_event'].iloc[0]

bills_df['bill_id_new'] = bills_df['bill_id_new_event']


#%%

parties_df = parties_df.merge(bills_df[['bill_id_new', 'bill_number', 'session_id']], on = ['bill_number', 'session_id'], how = 'left')

event_table =  event_table.merge(parties_df[['bill_id_new', 'is_matched_bill']].drop_duplicates(), on = 'bill_id_new', how = 'left')
event_table.to_csv(machine + 'tables/' + 'event_table.csv')

committee_members_df = committee_members_df.loc[~pd.isna(committee_members_df.committee_id),:]
committee_members_df_grouped = committee_members_df.groupby(['people_id_inferred', 'session_id'])['committee_id'].agg(lambda x: list(set(filter(lambda y: pd.notna(y), map(int, x))))).reset_index()

# Rename the columns as needed
committee_members_df_grouped.columns = ['people_id', 'session_id', 'people_id_committees']

parties_df = parties_df.merge(people_df[['people_id', 'session_id', 'party_id', 'role_id']], on = ['people_id', 'session_id'], how = 'left')
parties_df = parties_df.merge(committee_members_df_grouped, on = ['people_id', 'session_id'], how = 'left')
parties_df_temp = parties_df.loc[parties_df.is_matched_bill == True,['year', 'session_id', 'filing_bimonth', 'bill_number', 'bill_id_new']].drop_duplicates(subset = ['year', 'session_id', 'filing_bimonth', 'bill_number'])
#parties_df_temp = parties_df_temp.loc[parties_df_temp.is_matched_bill == True,:]
contact_table = event_table.merge(parties_df_temp, on = ['year', 'session_id', 'filing_bimonth', 'bill_id_new'], suffixes=('_event', '_contact'), how = 'outer')
contact_table['earliest_event_year'] = contact_table['year']
contact_table['earliest_event_bimonth'] = contact_table['filing_bimonth']
contact_table['earliest_event_date'] = contact_table['date']
contact_table['earliest_event_type'] = contact_table['event']
contact_table['contact_event_future_inferred'] = False
contact_table['contact_event_past_inferred'] = False

contact_grouped = contact_table.groupby('bill_id_new')

i=0
for name, group in contact_grouped:
    
    if i%100==0:
        print(i)
    # Step 2: Find rows where 'bill_number' is not missing and 'event' is missing
    subset = group[(~group['bill_number_contact'].isna()) & (group['event'].isna())]
    subset_event = group[~group['event'].isna()]
    max_year = group['year'].max()
    max_bimonth = group[group['year'] == max_year]['filing_bimonth'].max()

    inference_future = False
    inference_past = False
    if not subset.empty:
        # Step 3: Find the 'year' and 'filing_bimonth' for each matching row
        
        for index, row in subset.iterrows():
            
            year = row['year']
            bimonth = row['filing_bimonth']

            # Step 4: Find the earliest (year, filing_bimonth) that meets the criteria
            #next_bimonth = bimonth + 1
        
            for y in range(year, max_year + 1):
                
                if y == year:
                    flag_subset = subset_event.loc[subset_event.year == y, 'filing_bimonth']
                    if sum(flag_subset > bimonth) > 0:
                        new_bimonth = flag_subset[flag_subset > bimonth].min()
                        new_year = y
                        new_event = subset_event.loc[(subset_event.year == new_year) & (subset_event.filing_bimonth == new_bimonth), 'event'].iloc[0].astype(int)
                        new_date = subset_event.loc[(subset_event.year == new_year) & (subset_event.filing_bimonth == new_bimonth), 'date'].iloc[0]
                        inference_future = True
                        break
                    
                    
                        
                else:
                    flag_subset = subset_event.loc[subset_event.year == y, 'filing_bimonth']
                    if len(flag_subset > 0):
                        new_bimonth = flag_subset.min()
                        new_year = y
                        new_event = subset_event.loc[(subset_event.year == new_year) & (subset_event.filing_bimonth == new_bimonth), 'event'].iloc[0].astype(int)
                        new_date = subset_event.loc[(subset_event.year == new_year) & (subset_event.filing_bimonth == new_bimonth), 'date'].iloc[0]
                        inference_future = True
                        break
                    
            if inference_future == False:
                
                for y in range(year, max_year + 1):
                    
                    if y == year:
                        flag_subset = subset_event.loc[subset_event.year == y, 'filing_bimonth']
                        if sum(flag_subset < bimonth) > 0:
                            new_bimonth = flag_subset[flag_subset < bimonth].max()
                            new_year = y
                            new_event = subset_event.loc[(subset_event.year == new_year) & (subset_event.filing_bimonth == new_bimonth), 'event'].iloc[-1].astype(int)
                            new_date = subset_event.loc[(subset_event.year == new_year) & (subset_event.filing_bimonth == new_bimonth), 'date'].iloc[-1]
                            inference_past = True
                            break
                        
                        
                            
                    else:
                        flag_subset = subset_event.loc[subset_event.year == y, 'filing_bimonth']
                        if len(flag_subset > 0):
                            new_bimonth = flag_subset.max()
                            new_year = y
                            new_event = subset_event.loc[(subset_event.year == new_year) & (subset_event.filing_bimonth == new_bimonth), 'event'].iloc[-1].astype(int)
                            new_date = subset_event.loc[(subset_event.year == new_year) & (subset_event.filing_bimonth == new_bimonth), 'date'].iloc[-1]
                            inference_past = True
                            break
            
            contact_table.loc[index, 'earliest_event_year'] = new_year
            contact_table.loc[index, 'earliest_event_bimonth'] = new_bimonth
            contact_table.loc[index, 'earliest_event_date'] = new_date
            contact_table.loc[index, 'earliest_event_type'] = new_event
            contact_table.loc[index, 'contact_event_future_inferred'] = inference_future
            contact_table.loc[index, 'contact_event_past_inferred'] = inference_past
    i = i+1
    
    


#%%

contact_table_temp = contact_table[['session_id', 'bill_id_new', 'bill_number_contact', 'is_matched_bill', 'earliest_event_year', 'earliest_event_bimonth', 'earliest_event_type', 'earliest_event_date',  'contact_event_future_inferred', 'contact_event_past_inferred']]
contact_table_temp['earliest_event_date'] = contact_table_temp.earliest_event_date.apply(pd.Timestamp)
contact_table_temp = event_table.merge(contact_table_temp, left_on = ['session_id', 'bill_id_new', 'year', 'filing_bimonth', 'event', 'date'], right_on = ['session_id', 'bill_id_new', 'earliest_event_year', 'earliest_event_bimonth', 'earliest_event_type', 'earliest_event_date'], suffixes=('_event', '_contact'), how = 'right')

contact_table = contact_table_temp.merge(parties_df.loc[parties_df.is_matched_bill == True,:], left_on = ['session_id', 'bill_id_new', 'bill_number_contact', 'year', 'filing_bimonth' ], right_on = ['session_id', 'bill_id_new', 'bill_number',  'year', 'filing_bimonth'],  suffixes=('_event', '_contact'), how = 'right')

#contact_table.to_csv(machine + 'tables/' + 'merged_legislation_contacts_outer.csv')

#%%

# Create a dataframe to show the history of contacts

def check_people_sponsor(people_id, sponsors):
    flag = False
    for s in sponsors:
        if people_id == s[0]:
            flag = True
            break
        
    return flag

def check_committee(committee_id, committees):
    flag = False
    if type(committees) == list:
        if committee_id in committees:
            flag = True
    else:
        if committee_id == committees:
            flag = True
        
    return flag

def check_committee_member(committee_people_id, committees):
 
    flag = False
    if type(committees) == list:
        if len(set(committee_people_id) & set(committees))>0:
            flag = True
    else:
        if committees in committee_people_id:
            flag = True
        
    return flag


    
    
contact_table['is_sponsors_chamber_1_type_1'] = contact_table.loc[~contact_table.sponsors_chamber_1_type_1.isna() & (~contact_table.people_id.isna()), ['people_id', 'sponsors_chamber_1_type_1']].apply(lambda row: check_people_sponsor(row['people_id'], row['sponsors_chamber_1_type_1']), axis=1)
contact_table['is_sponsors_chamber_1_type_2'] = contact_table.loc[~contact_table.sponsors_chamber_1_type_2.isna() & (~contact_table.people_id.isna()), ['people_id', 'sponsors_chamber_1_type_2']].apply(lambda row: check_people_sponsor(row['people_id'], row['sponsors_chamber_1_type_2']), axis=1)
contact_table['is_sponsors_chamber_2_type_1'] = contact_table.loc[~contact_table.sponsors_chamber_2_type_1.isna() & (~contact_table.people_id.isna()), ['people_id', 'sponsors_chamber_2_type_1']].apply(lambda row: check_people_sponsor(row['people_id'], row['sponsors_chamber_2_type_1']), axis=1)
contact_table['is_sponsors_chamber_2_type_2'] = contact_table.loc[~contact_table.sponsors_chamber_2_type_2.isna() & (~contact_table.people_id.isna()), ['people_id', 'sponsors_chamber_2_type_2']].apply(lambda row: check_people_sponsor(row['people_id'], row['sponsors_chamber_2_type_2']), axis=1)

contact_table['is_chair_committee_chamber_1_type_1'] = contact_table['people_id'] == contact_table['chair_committee_chamber_1_type_1']

contact_table['is_committee_chamber_1_type_1'] = contact_table.loc[~contact_table.committee_chamber_1_type_1.isna() & (~contact_table.committee_id.isna()), ['committee_id', 'committee_chamber_1_type_1']].apply(lambda row: check_committee(row['committee_id'], row['committee_chamber_1_type_1']), axis=1)
contact_table['is_committee_chamber_1_type_2'] = contact_table.loc[~contact_table.committee_chamber_1_type_2.isna() & (~contact_table.committee_id.isna()), ['committee_id', 'committee_chamber_1_type_2']].apply(lambda row: check_committee(row['committee_id'], row['committee_chamber_1_type_2']), axis=1)
contact_table['is_committee_chamber_2_type_1'] = contact_table.loc[~contact_table.committee_chamber_2_type_1.isna() & (~contact_table.committee_id.isna()), ['committee_id', 'committee_chamber_2_type_1']].apply(lambda row: check_committee(row['committee_id'], row['committee_chamber_2_type_1']), axis=1)
contact_table['is_committee_chamber_2_type_2'] = contact_table.loc[~contact_table.committee_chamber_2_type_2.isna() & (~contact_table.committee_id.isna()), ['committee_id', 'committee_chamber_2_type_2']].apply(lambda row: check_committee(row['committee_id'], row['committee_chamber_2_type_2']), axis=1)

contact_table['is_committee_member_chamber_1_type_1'] = contact_table.loc[(~contact_table.committee_chamber_1_type_1.isna()) & (~contact_table.people_id.isna()) & (~contact_table.people_id_committees.isna()), ['people_id_committees', 'committee_chamber_1_type_1']].apply(lambda row: check_committee_member(row['people_id_committees'], row['committee_chamber_1_type_1']), axis=1)
contact_table['is_committee_member_chamber_1_type_2'] = contact_table.loc[(~contact_table.committee_chamber_1_type_2.isna()) & (~contact_table.people_id.isna()) & (~contact_table.people_id_committees.isna()), ['people_id_committees', 'committee_chamber_1_type_2']].apply(lambda row: check_committee_member(row['people_id_committees'], row['committee_chamber_1_type_2']), axis=1)
contact_table['is_committee_member_chamber_2_type_1'] = contact_table.loc[(~contact_table.committee_chamber_2_type_1.isna()) & (~contact_table.people_id.isna()) & (~contact_table.people_id_committees.isna()), ['people_id_committees', 'committee_chamber_2_type_1']].apply(lambda row: check_committee_member(row['people_id_committees'], row['committee_chamber_2_type_1']), axis=1)
contact_table['is_committee_member_chamber_2_type_2'] = contact_table.loc[(~contact_table.committee_chamber_2_type_2.isna()) & (~contact_table.people_id.isna()) & (~contact_table.people_id_committees.isna()), ['people_id_committees', 'committee_chamber_2_type_2']].apply(lambda row: check_committee_member(row['people_id_committees'], row['committee_chamber_2_type_2']), axis=1)

contact_table['is_chamber_1'] = contact_table['chamber_1'] == contact_table['chamber_id_contact']
contact_table['is_chamber_2'] = contact_table['chamber_2'] == contact_table['chamber_id_contact']
contact_table['is_member_chamber_1'] = contact_table['is_chamber_1'] & pd.isna(contact_table['people_id'])
contact_table['is_member_chamber_2'] = contact_table['is_chamber_2'] & pd.isna(contact_table['people_id'])
contact_table['is_member'] = pd.isna(contact_table['people_id'])


cols_to_fill = ['is_sponsors_chamber_1_type_1',
                'is_sponsors_chamber_1_type_2',
                'is_sponsors_chamber_2_type_1',
                'is_sponsors_chamber_2_type_2',
                'is_chair_committee_chamber_1_type_1',
                'is_committee_member_chamber_1_type_1',
                'is_committee_member_chamber_1_type_2',
                'is_committee_member_chamber_2_type_1',
                'is_committee_member_chamber_2_type_2',
                'is_committee_chamber_1_type_1',
                'is_committee_chamber_1_type_2',
                'is_committee_chamber_2_type_1',
                'is_committee_chamber_2_type_2',
                'is_member_chamber_1',
                'is_member_chamber_2',
                'is_chamber_1',
                'is_chamber_2',
                'is_member'
                ]

contact_table[cols_to_fill] = contact_table[cols_to_fill].fillna(False)
contact_table['is_sponsor_chamber_1'] = contact_table['is_sponsors_chamber_1_type_1'] | contact_table['is_sponsors_chamber_1_type_2']
contact_table['is_sponsor_chamber_2'] = contact_table['is_sponsors_chamber_2_type_1'] | contact_table['is_sponsors_chamber_2_type_2']
contact_table['is_sponsor_contact'] = contact_table['is_sponsor_chamber_1'] | contact_table['is_sponsor_chamber_2']

contact_table['is_committee_chamber_1'] = contact_table['is_committee_chamber_1_type_1'] | contact_table['is_committee_chamber_1_type_2']
contact_table['is_committee_chamber_2'] = contact_table['is_committee_chamber_2_type_1'] | contact_table['is_committee_chamber_2_type_2']
contact_table['is_committee_contact'] = contact_table['is_committee_chamber_1'] | contact_table['is_committee_chamber_2']

contact_table['is_committee_member_chamber_1_type_1_excluding_sponsors'] = contact_table['is_committee_member_chamber_1_type_1'] & ~contact_table['is_sponsor_chamber_1']
contact_table['is_committee_member_chamber_1_type_2_excluding_sponsors'] = contact_table['is_committee_member_chamber_1_type_2'] & ~contact_table['is_sponsor_chamber_1']
contact_table['is_committee_member_chamber_2_type_1_excluding_sponsors'] = contact_table['is_committee_member_chamber_2_type_1'] & ~contact_table['is_sponsor_chamber_2']
contact_table['is_committee_member_chamber_2_type_2_excluding_sponsors'] = contact_table['is_committee_member_chamber_2_type_2'] & ~contact_table['is_sponsor_chamber_2']

contact_table['is_member_chamber_1_excluding_sponsors_committee_members'] = contact_table['is_member_chamber_1'] & ~(contact_table['is_sponsor_chamber_1'] | contact_table['is_committee_chamber_1'])
contact_table['is_member_chamber_2_excluding_sponsors_committee_members'] = contact_table['is_member_chamber_2'] & ~(contact_table['is_sponsor_chamber_2'] | contact_table['is_committee_chamber_2'])
#%%

# write the history of bills
event_grouped = event_table.groupby(['bill_id_new','event', 'date'])

# write the history of contacts
contact_grouped = contact_table.groupby(['bill_id_new','event', 'date'])

contact_grouped = contact_grouped.agg(
    total_contacts=pd.NamedAgg(column='bill_id_new', aggfunc='count'),
    total_member_contacts = pd.NamedAgg(column='is_member', aggfunc=lambda x: x.dropna().sum()),
    sum_sponsors_chamber_1_type_1=pd.NamedAgg(column='is_sponsors_chamber_1_type_1', aggfunc=lambda x: x.dropna().sum()),
    sum_sponsors_chamber_1_type_2=pd.NamedAgg(column='is_sponsors_chamber_1_type_2', aggfunc=lambda x: x.dropna().sum()),
    sum_sponsors_chamber_2_type_1=pd.NamedAgg(column='is_sponsors_chamber_2_type_1', aggfunc=lambda x: x.dropna().sum()),
    sum_sponsors_chamber_2_type_2=pd.NamedAgg(column='is_sponsors_chamber_2_type_2', aggfunc=lambda x: x.dropna().sum()),
    sum_chair_committee_chamber_1_type_1=pd.NamedAgg(column='is_chair_committee_chamber_1_type_1', aggfunc=lambda x: x.dropna().sum()),
    sum_committee_members_chamber_1_type_1=pd.NamedAgg(column='is_committee_member_chamber_1_type_1', aggfunc=lambda x: x.dropna().sum()),
    sum_committee_members_chamber_1_type_2=pd.NamedAgg(column='is_committee_member_chamber_1_type_2', aggfunc=lambda x: x.dropna().sum()),
    sum_committee_members_chamber_2_type_1=pd.NamedAgg(column='is_committee_member_chamber_2_type_1', aggfunc=lambda x: x.dropna().sum()),
    sum_committee_members_chamber_2_type_2=pd.NamedAgg(column='is_committee_member_chamber_2_type_2', aggfunc=lambda x: x.dropna().sum()),
    sum_committee_members_chamber_1_type_1_excluding_sponsors=pd.NamedAgg(column='is_committee_member_chamber_1_type_1_excluding_sponsors', aggfunc=lambda x: x.dropna().sum()),
    sum_committee_members_chamber_1_type_2_excluding_sponsors=pd.NamedAgg(column='is_committee_member_chamber_1_type_2_excluding_sponsors', aggfunc=lambda x: x.dropna().sum()),
    sum_committee_members_chamber_2_type_1_excluding_sponsors=pd.NamedAgg(column='is_committee_member_chamber_2_type_1_excluding_sponsors', aggfunc=lambda x: x.dropna().sum()),
    sum_committee_members_chamber_2_type_2_excluding_sponsors=pd.NamedAgg(column='is_committee_member_chamber_2_type_2_excluding_sponsors', aggfunc=lambda x: x.dropna().sum()),
    sum_committee_chamber_1_type_1=pd.NamedAgg(column='is_committee_chamber_1_type_1', aggfunc=lambda x: x.dropna().sum()),
    sum_committee_chamber_1_type_2=pd.NamedAgg(column='is_committee_chamber_1_type_2', aggfunc=lambda x: x.dropna().sum()),
    sum_committee_chamber_2_type_1=pd.NamedAgg(column='is_committee_chamber_2_type_1', aggfunc=lambda x: x.dropna().sum()),
    sum_committee_chamber_2_type_2=pd.NamedAgg(column='is_committee_chamber_2_type_2', aggfunc=lambda x: x.dropna().sum()),
    sum_member_chamber_1_excluding_sponsors_committee_members=pd.NamedAgg(column='is_member_chamber_1_excluding_sponsors_committee_members', aggfunc=lambda x: x.dropna().sum()),
    sum_member_chamber_2_excluding_sponsors_committee_members=pd.NamedAgg(column='is_member_chamber_2_excluding_sponsors_committee_members', aggfunc=lambda x: x.dropna().sum()),
    sum_member_chamber_1=pd.NamedAgg(column='is_member_chamber_1', aggfunc=lambda x: x.dropna().sum()),
    sum_member_chamber_2=pd.NamedAgg(column='is_member_chamber_2', aggfunc=lambda x: x.dropna().sum()),
    sum_is_chamber_1=pd.NamedAgg(column='is_chamber_1', aggfunc=lambda x: x.dropna().sum()),
    sum_is_chamber_2=pd.NamedAgg(column='is_chamber_2', aggfunc=lambda x: x.dropna().sum())
    
).reset_index()

contact_grouped.to_csv(machine + 'tables/' + 'contact_grouped.csv')
#%%


sponsors_table = contact_table.loc[contact_table.is_committee_chamber_1_type_1 == True,:]

contact_grouped = contact_table.groupby(['bill_id_new'])


earliest_year = contact_table.groupby(['bill_id_new']).year.min()
contact_table = contact_table.merge(earliest_year.rename('earliest_year'), on = 'bill_id_new', how = 'left')
earliest_bimonth = contact_table.loc[contact_table.earliest_year == contact_table.year,:].groupby(['bill_id_new']).filing_bimonth.min()
contact_table = contact_table.merge(earliest_bimonth.rename('earliest_bimonth'), on = 'bill_id_new', how = 'left')

def count_bimonths(row):
    
    length = 0
    if row['year'] == row['earliest_year']:
        
        length = row['filing_bimonth'] - row['earliest_bimonth']
        
    elif row['year'] > row['earliest_year']:
        
        length = 6 - row['earliest_bimonth']
        
        for y in range(int(row['year']-row['earliest_year'])):
            
            if row['year'] > (row['earliest_year'] + 1):
                
                length = length + 6
                
            else:
                
                length = length + row['filing_bimonth']
                
                
    return length

contact_table['event_bimonth_sequence'] = contact_table.apply(count_bimonths, axis= 1)


sponsors_table = contact_table.loc[contact_table.is_sponsors_chamber_1_type_1 == True,:]

sponsors_table = sponsors_table.merge(parties_df.loc[(parties_df.is_matched_bill == False) & (~parties_df.people_id.isna()), ['year', 'filing_bimonth', 'people_id', 'principal_lobbyist', 'beneficial_client']].drop_duplicates(), suffixes = ('', '_unmatched'), on = ['people_id', 'principal_lobbyist', 'beneficial_client'], how='left')
sponsors_table = sponsors_table.merge(parties_df.loc[(parties_df.is_matched_bill == True) & (~parties_df.people_id.isna()), ['year', 'filing_bimonth', 'people_id', 'principal_lobbyist', 'beneficial_client']].drop_duplicates(), suffixes = ('', '_same'), on = ['people_id', 'principal_lobbyist', 'beneficial_client'], how='left')

sponsors_table['diff_year_unmatched'] = sponsors_table.year_unmatched-sponsors_table.earliest_year
sponsors_table['diff_bimonth_unmatched'] = sponsors_table.filing_bimonth_unmatched - sponsors_table.earliest_bimonth

cond = ((sponsors_table.diff_year_unmatched==0) & (sponsors_table.diff_bimonth_unmatched>0)) | (sponsors_table.diff_year_unmatched>0)

sponsors_table_temp = sponsors_table.loc[cond,:].copy()

sponsors_table_temp.loc[sponsors_table_temp.diff_year_unmatched==0, 'length_before_intro'] = -sponsors_table_temp.loc[sponsors_table_temp.diff_year_unmatched==0,'diff_bimonth_unmatched']
sponsors_table_temp.loc[sponsors_table_temp.diff_year_unmatched>0, 'length_before_intro'] = -(sponsors_table_temp.loc[sponsors_table_temp.diff_year_unmatched>0,'earliest_bimonth'] - (6-sponsors_table_temp.loc[sponsors_table_temp.diff_year_unmatched>0,'filing_bimonth_unmatched']))
sponsors_table_temp = sponsors_table_temp.loc[(sponsors_table_temp.length_before_intro < 0) & (sponsors_table_temp.length_before_intro >= -6)]
sponsors_table_temp = sponsors_table_temp.loc[:,['event_bimonth_sequence', 'length_before_intro', 'people_id', 'principal_lobbyist', 'beneficial_client']]

sponsors_table_new = sponsors_table.merge(sponsors_table_temp.drop_duplicates(), on = ['people_id', 'principal_lobbyist', 'beneficial_client'], how='left')
sponsors_table_new = sponsors_table_new.loc[(sponsors_table_new.length_before_intro < 0) & (sponsors_table_new.length_before_intro >= -6)]


sponsors_table['diff_year_same'] = sponsors_table.year_same-sponsors_table.earliest_year
sponsors_table['diff_bimonth_same'] = sponsors_table.filing_bimonth_same - sponsors_table.earliest_bimonth

cond = ((sponsors_table.diff_year_same==0) & (sponsors_table.diff_bimonth_same>0)) | (sponsors_table.diff_year_same>0)

sponsors_table_temp = sponsors_table.loc[cond,:].copy()

sponsors_table_temp.loc[sponsors_table_temp.diff_year_same==0, 'length_before_intro_same'] = -sponsors_table_temp.loc[sponsors_table_temp.diff_year_same==0,'diff_bimonth_same']
sponsors_table_temp.loc[sponsors_table_temp.diff_year_same>0, 'length_before_intro_same'] = -(sponsors_table_temp.loc[sponsors_table_temp.diff_year_same>0,'earliest_bimonth'] - (6-sponsors_table_temp.loc[sponsors_table_temp.diff_year_same>0,'filing_bimonth_same']))
sponsors_table_temp = sponsors_table_temp.loc[(sponsors_table_temp.length_before_intro_same < 0) & (sponsors_table_temp.length_before_intro_same >= -6)]
sponsors_table_temp = sponsors_table_temp.loc[:,['event_bimonth_sequence', 'length_before_intro_same', 'people_id', 'principal_lobbyist', 'beneficial_client']]

sponsors_table_new_same = sponsors_table.merge(sponsors_table_temp.drop_duplicates(subset = ['people_id', 'principal_lobbyist', 'beneficial_client','length_before_intro_same']), on = ['people_id', 'principal_lobbyist', 'beneficial_client'], how='left')
sponsors_table_new_same = sponsors_table_new_same.loc[(sponsors_table_new_same.length_before_intro_same < 0) & (sponsors_table_new.length_before_intro_same >= -6)]


sponsors_grouped = sponsors_table.groupby(['bill_id_new', 'event_bimonth_sequence'])

total_lobbyists = contact_table.groupby('bill_id_new').principal_lobbyist.nunique().reset_index()

sponsor_agg = sponsors_grouped.agg(
    total_contacts=pd.NamedAgg(column='bill_id_new', aggfunc='count'),
    distinct_lobbyists = pd.NamedAgg(column='principal_lobbyist', aggfunc='nunique'),
    distinct_retained_lobbyists = pd.NamedAgg(column='principal_lobbyist', aggfunc=lambda x: x[x.isin(sponsors_table[sponsors_table['retained_flag'] == True]['principal_lobbyist'].unique())].nunique()),
    distinct_employed_lobbyists = pd.NamedAgg(column='principal_lobbyist', aggfunc=lambda x: x[x.isin(sponsors_table[sponsors_table['retained_flag'] == False]['principal_lobbyist'].unique())].nunique())
    ).reset_index()

sponsor_agg['class_agg'] = 'exact_bill'
sponsors_new_grouped = sponsors_table_new.groupby(['bill_id_new', 'length_before_intro'])

sponsor_new_agg = sponsors_new_grouped.agg(
    total_contacts=pd.NamedAgg(column='bill_id_new', aggfunc='count'),
    distinct_lobbyists = pd.NamedAgg(column='principal_lobbyist', aggfunc='nunique'),
    distinct_retained_lobbyists = pd.NamedAgg(column='principal_lobbyist', aggfunc=lambda x: x[x.isin(sponsors_table[sponsors_table['retained_flag'] == True]['principal_lobbyist'].unique())].nunique()),
    distinct_employed_lobbyists = pd.NamedAgg(column='principal_lobbyist', aggfunc=lambda x: x[x.isin(sponsors_table[sponsors_table['retained_flag'] == False]['principal_lobbyist'].unique())].nunique())

    ).reset_index()

sponsor_new_agg.rename({'length_before_intro': 'event_bimonth_sequence'}, axis=1,inplace=True)
sponsor_agg['class_agg'] = 'no_id'

sponsors_new_same_grouped = sponsors_table_new_same.groupby(['bill_id_new', 'length_before_intro'])

sponsor_new_agg_same = sponsors_new_same_grouped.agg(
    total_contacts=pd.NamedAgg(column='bill_id_new', aggfunc='count'),
    distinct_lobbyists = pd.NamedAgg(column='principal_lobbyist', aggfunc='nunique'),
    distinct_retained_lobbyists = pd.NamedAgg(column='principal_lobbyist', aggfunc=lambda x: x[x.isin(sponsors_table[sponsors_table['retained_flag'] == True]['principal_lobbyist'].unique())].nunique()),
    distinct_employed_lobbyists = pd.NamedAgg(column='principal_lobbyist', aggfunc=lambda x: x[x.isin(sponsors_table[sponsors_table['retained_flag'] == False]['principal_lobbyist'].unique())].nunique())

    ).reset_index()

sponsor_new_agg_same.rename({'length_before_intro': 'event_bimonth_sequence'}, axis=1,inplace=True)
sponsor_agg_same['class_agg'] = 'before_id'


sponsors_contacts_table = pd.concat((sponsor_agg, sponsor_new_agg), axis=0)
sponsors_contacts_table = pd.concat((sponsors_contacts_table, sponsor_new_agg_same), axis=0)
sponsors_contacts_table = sponsors_contacts_table.merge(total_lobbyists, on = 'bill_id_new', how = 'left')
sponsors_contacts_table.to_csv(machine + 'tables/' + 'sponsors_contacts_table.csv')

#%%

sponsor_table_agg = sponsors_table.copy()
cond = ((sponsors_table.diff_year_unmatched==0) & (sponsors_table.diff_bimonth_unmatched>0)) | (sponsors_table.diff_year_unmatched>0)

sponsor_table_agg.loc[cond & sponsor_table_agg.diff_year_unmatched==0, 'length_before_intro'] = sponsor_table_agg.loc[cond & sponsor_table_agg.diff_year_unmatched==0,'diff_bimonth_unmatched']
sponsor_table_agg.loc[cond & sponsor_table_agg.diff_year_unmatched<0, 'length_before_intro'] = -(sponsor_table_agg.loc[cond & sponsor_table_agg.diff_year_unmatched<0,'earliest_bimonth'] + (6-sponsor_table_agg.loc[cond & sponsor_table_agg.diff_year_unmatched<0,'filing_bimonth_unmatched']))
sponsor_table_agg = sponsor_table_agg[(sponsor_table_agg['earliest_year'] > 2019) | ((sponsor_table_agg['earliest_year'] == 2019) & (sponsor_table_agg['earliest_bimonth'] > 3))]

total_lobbyists.rename({'principal_lobbyist':'total_lobbyists'}, inplace=True, axis=1)
sponsor_table_agg = sponsor_table_agg.merge(total_lobbyists, on = 'bill_id_new', how='left')
sponsor_table_agg.loc[~pd.isna(sponsor_table_agg.length_before_intro),'event_bimonth_sequence'] = sponsor_table_agg.loc[~pd.isna(sponsor_table_agg.length_before_intro),'length_before_intro']
sponsor_table_agg_ = sponsor_table_agg.loc[sponsor_table_agg.total_lobbyists<=4,:]

positive_events = sponsor_table_agg_[sponsor_table_agg_['event_bimonth_sequence'] >= 0]

# Count occurrences for each (bill_id_new, principal_lobbyist)
event_counts = positive_events.groupby(['bill_id_new', 'principal_lobbyist']).size().reset_index(name='event_counts')
event_counts['event_counts_mod'] = event_counts.event_counts + np.random.rand(len(event_counts))
# Sort within each bill_id_new in descending order by the event_counts

event_counts['rank'] = event_counts.groupby('bill_id_new')['event_counts_mod'].rank(ascending=False, method='min')

lobbyist_ranks = event_counts.pivot(index='bill_id_new', columns='rank', values='principal_lobbyist').reset_index()

# Left merge this new table to sponsor_table_agg_
merged_table = pd.merge(sponsor_table_agg_, lobbyist_ranks, on='bill_id_new', how='left')
merged_table['contacts_1'] = merged_table['principal_lobbyist'] == merged_table[1]
merged_table['contacts_2'] = merged_table['principal_lobbyist'] == merged_table[2]
merged_table['contacts_3'] = merged_table['principal_lobbyist'] == merged_table[3]
merged_table['contacts_4'] = merged_table['principal_lobbyist'] == merged_table[4]
# Group by each bill_id, event_bi_month sequence, and count the contacts made by each lobbyist
contacts_sum = merged_table.groupby(['event_bimonth_sequence', 'total_lobbyists']).agg({
    'contacts_1': 'sum',
    'contacts_2': 'sum',
    'contacts_3': 'sum',
    'contacts_4': 'sum'
}).reset_index()

contacts_avg = merged_table.groupby(['event_bimonth_sequence', 'total_lobbyists']).agg({
    'contacts_1': 'sum',
    'contacts_2': 'sum',
    'contacts_3': 'sum',
    'contacts_4': 'sum',
    'bill_id_new': 'count'  # Count the number of unique bills
}).reset_index()

# Calculate average contacts per bill for each lobbyist
contacts_avg['avg_contacts_1'] = contacts_avg['contacts_1'] / contacts_avg['bill_id_new']
contacts_avg['avg_contacts_2'] = contacts_avg['contacts_2'] / contacts_avg['bill_id_new']
contacts_avg['avg_contacts_3'] = contacts_avg['contacts_3'] / contacts_avg['bill_id_new']
contacts_avg['avg_contacts_4'] = contacts_avg['contacts_4'] / contacts_avg['bill_id_new']

contacts_sum.to_csv(machine + 'tables/' + 'contacts_history.csv')
contacts_avg.to_csv(machine + 'tables/' + 'contacts_history_avg.csv')



cond = ((sponsors_table.diff_year_same==0) & (sponsors_table.diff_bimonth_same>0)) | (sponsors_table.diff_year_same>0)


sponsor_table_agg = sponsors_table.copy()
sponsor_table_agg.loc[cond & sponsor_table_agg.diff_year_same==0, 'length_before_intro_same'] = sponsor_table_agg.loc[cond & sponsor_table_agg.diff_bimonth_same==0,'diff_bimonth_same']
sponsor_table_agg.loc[cond & sponsor_table_agg.diff_year_same<0, 'length_before_intro_same'] = -(sponsor_table_agg.loc[cond & sponsor_table_agg.diff_bimonth_same<0,'earliest_bimonth'] + (6-sponsor_table_agg.loc[cond & sponsor_table_agg.diff_year_same<0,'filing_bimonth_same']))
sponsor_table_agg = sponsor_table_agg[(sponsor_table_agg['earliest_year'] > 2019) | ((sponsor_table_agg['earliest_year'] == 2019) & (sponsor_table_agg['earliest_bimonth'] > 3))]

total_lobbyists.rename({'principal_lobbyist':'total_lobbyists'}, inplace=True, axis=1)
sponsor_table_agg = sponsor_table_agg.merge(total_lobbyists, on = 'bill_id_new', how='left')
sponsor_table_agg.loc[~pd.isna(sponsor_table_agg.length_before_intro_same),'event_bimonth_sequence'] = sponsor_table_agg.loc[~pd.isna(sponsor_table_agg.length_before_intro_same),'length_before_intro_same']
sponsor_table_agg_ = sponsor_table_agg.loc[sponsor_table_agg.total_lobbyists<=4,:]

positive_events = sponsor_table_agg_[sponsor_table_agg_['event_bimonth_sequence'] >= 0]

# Count occurrences for each (bill_id_new, principal_lobbyist)
event_counts = positive_events.groupby(['bill_id_new', 'principal_lobbyist']).size().reset_index(name='event_counts')
event_counts['event_counts_mod'] = event_counts.event_counts + np.random.rand(len(event_counts))
# Sort within each bill_id_new in descending order by the event_counts

event_counts['rank'] = event_counts.groupby('bill_id_new')['event_counts_mod'].rank(ascending=False, method='min')

lobbyist_ranks = event_counts.pivot(index='bill_id_new', columns='rank', values='principal_lobbyist').reset_index()

# Left merge this new table to sponsor_table_agg_
merged_table = pd.merge(sponsor_table_agg_, lobbyist_ranks, on='bill_id_new', how='left')
merged_table['contacts_1'] = merged_table['principal_lobbyist'] == merged_table[1]
merged_table['contacts_2'] = merged_table['principal_lobbyist'] == merged_table[2]
merged_table['contacts_3'] = merged_table['principal_lobbyist'] == merged_table[3]
merged_table['contacts_4'] = merged_table['principal_lobbyist'] == merged_table[4]
# Group by each bill_id, event_bi_month sequence, and count the contacts made by each lobbyist
contacts_sum = merged_table.groupby(['event_bimonth_sequence', 'total_lobbyists']).agg({
    'contacts_1': 'sum',
    'contacts_2': 'sum',
    'contacts_3': 'sum',
    'contacts_4': 'sum'
}).reset_index()

contacts_avg = merged_table.groupby(['event_bimonth_sequence', 'total_lobbyists']).agg({
    'contacts_1': 'sum',
    'contacts_2': 'sum',
    'contacts_3': 'sum',
    'contacts_4': 'sum',
    'bill_id_new': 'count'  # Count the number of unique bills
}).reset_index()

# Calculate average contacts per bill for each lobbyist
contacts_avg['avg_contacts_1'] = contacts_avg['contacts_1'] / contacts_avg['bill_id_new']
contacts_avg['avg_contacts_2'] = contacts_avg['contacts_2'] / contacts_avg['bill_id_new']
contacts_avg['avg_contacts_3'] = contacts_avg['contacts_3'] / contacts_avg['bill_id_new']
contacts_avg['avg_contacts_4'] = contacts_avg['contacts_4'] / contacts_avg['bill_id_new']

contacts_sum.to_csv(machine + 'tables/' + 'contacts_history_new.csv')
contacts_avg.to_csv(machine + 'tables/' + 'contacts_history_avg_new.csv')


# Step 2: Merge with sponsor_table_agg_
merged_result = pd.merge(contacts_sum, sponsor_table_agg_[['bill_id_new', 'earliest_year', 'earliest_bimonth']].drop_duplicates(subset='bill_id_new'), on='bill_id_new', how='left')

# Step 4: Filter for the desired conditions
filtered_result = merged_result[(merged_result['earliest_year'] > 2019) | ((merged_result['earliest_year'] == 2019) & (merged_result['earliest_bimonth'] > 3))]

#%%
def fillna_locf(series):
    return series.fillna(method='ffill')

# Create chamber_id_filled
event_table['chamber_id_filled'] = np.where(
    (event_table['chamber_id'].isna() & (event_table['date_number'] == event_table.groupby('bill_id_new')['date_number'].transform('min'))),
    event_table['chamber_1'],
    fillna_locf(event_table['chamber_id'])
)

# Create committee_id_filled
event_table['committee_id_filled'] = np.where((event_table['event'] == 9) & event_table['committee_id_event'].isna(),
                                              fillna_locf(event_table['committee_id_event']),
                                              event_table['committee_id_event'])

# Create event_new column
event_table['event_new'] = event_table['event']

# Update event_new based on conditions
event_table.loc[(event_table['event'] == 9) &
                (event_table['chamber_id_filled'] == event_table['chamber_1']) &
                (event_table['committee_id_filled'] == event_table['committee_chamber_1_type_1']), 'event_new'] = 91

event_table.loc[(event_table['event'] == 9) &
                (event_table['chamber_id_filled'] == event_table['chamber_1']) &
                (event_table['committee_id_filled'] != event_table['committee_chamber_1_type_1']), 'event_new'] = 92

event_table.loc[(event_table['event'] == 9) &
                (event_table['chamber_id_filled'] != event_table['chamber_1']) &
                (event_table['committee_id_filled'] == event_table['committee_chamber_2_type_1']), 'event_new'] = 91

event_table.loc[(event_table['event'] == 9) &
                (event_table['chamber_id_filled'] != event_table['chamber_1']) &
                (event_table['committee_id_filled'] != event_table['committee_chamber_2_type_1']), 'event_new'] = 92

grouped_event_table = event_table.groupby('bill_id_new')

# Initialize columns
event_table['AIC'] = 0
event_table['ABC'] = 0
event_table['PASS'] = 0
event_table['LAW'] = 0

# Iterate over groups
for name, group in grouped_event_table:
    # Check if there is at least one event_new different from 1, 91, 92
    if not all(group['event_new'].isin([1, 91])):
        event_table.loc[group.index, 'AIC'] = 1

    # Check if there is at least one event different from 1, 91, 92
    # and where committee_id_filled != committee_chamber_1_type_1
    if any((group['event_new'] != 1) & (group['event_new'] != 91) & (group['event_new'] != 92) & (group['committee_id_filled'] != group['committee_chamber_1_type_1'])):
        event_table.loc[group.index, 'ABC'] = 1

    # Check if there is at least a 4
    if any(group['event_new'] == 10):
        event_table.loc[group.index, 'PASS'] = 1

    # Check if there is at least an 8
    if any(group['event_new'] == 8):
        event_table.loc[group.index, 'LAW'] = 1
        
        
event_table.loc[event_table.LAW == 1, 'PASS'] =1

grouped_event_table = event_table.groupby('bill_id_new').agg({
    'AIC': 'max',
    'ABC': 'max',
    'PASS': 'max',
    'LAW': 'max'
}).reset_index()

# Add INTRO column
grouped_event_table['INTRO'] = 1

bills_conditions = grouped_event_table.copy()
bills_conditions.to_csv(machine + 'tables/' + 'bills_conditions.csv')
#%%
find_lobbyist_author_1 = sponsors_table.groupby(['bill_id_new', 'beneficial_client', 'principal_lobbyist']).event_bimonth_sequence.nunique()
find_lobbyist_author_2 = sponsors_table_new.groupby(['bill_id_new', 'beneficial_client', 'principal_lobbyist']).length_before_intro.nunique()
find_lobbyist_author = pd.DataFrame(find_lobbyist_author_1).merge(pd.DataFrame(find_lobbyist_author_2), on = ['bill_id_new', 'beneficial_client', 'principal_lobbyist'], how = 'outer')
find_lobbyist_author['length_before_intro'] = find_lobbyist_author['length_before_intro'].fillna(0)
find_lobbyist_author['is_bill_lobbyist_lawmaker_matched'] = ((find_lobbyist_author.length_before_intro > 0) & (find_lobbyist_author.event_bimonth_sequence > 0)) #| (find_lobbyist_author.event_bimonth_sequence > 1)
find_lobbyist_author = find_lobbyist_author.reset_index()


matched_contact_bills = parties_df.loc[parties_df.is_matched_bill, ['bill_id_new', 'is_matched_bill']].drop_duplicates()
matched_authors_bills = find_lobbyist_author.groupby('bill_id_new').is_bill_lobbyist_lawmaker_matched.sum() > 0

sponsors_df = pd.read_csv(machine + legi_subfolder + 'sponsors_all.csv')
sponsors_df = sponsors_df[sponsors_df['session_id'].isin([1644, 1813])]
sponsors_df = sponsors_df.merge(people_df.loc[:, ['people_id', 'party_id', 'session_id', 'role_id']], on = ['people_id', 'session_id'], how = 'left')

bills_temp = event_table.loc[:, ['bill_id_new', 'bill_id', 'session_id', 'passed', 'chamber_1', 'chamber_2', 'committee_chamber_1_type_1', 'body_id', 'sponsors_chamber_1_type_1', 'date']].sort_values('date').groupby(['bill_id_new', 'bill_id', 'session_id'], as_index=False).first()
bills_temp['intro_year'] = bills_temp.date.dt.year
bills_temp['intro_month'] = bills_temp.date.dt.month
bills_temp['intro_semester'] = bills_temp.date.dt.month.apply(lambda x: round(x/12)+1)
bills_temp = bills_temp.loc[bills_temp.body_id == bills_temp.chamber_1,:]
authors_df = sponsors_df.merge(bills_temp, on = ['bill_id', 'session_id'], suffixes = ('_sponsor', '_bill'), how = 'inner')
authors_df = authors_df.loc[authors_df.sponsor_type_id==1,:]

authors_df_temp = pd.merge(authors_df, bills_conditions, on="bill_id_new")

# Select relevant columns
authors_df_temp = authors_df_temp[["people_id", "role_id", "session_id", "bill_id_new", "AIC", "ABC", "PASS", "LAW", "INTRO"]]

# Create a temporary table without duplicates
temp_table = authors_df_temp.drop_duplicates(subset=["session_id", "role_id", "bill_id_new", "AIC", "ABC", "PASS", "LAW", "INTRO"])

# Compute the sum of each condition for each session_id
sum_bills_table = temp_table.groupby("session_id")[["AIC", "ABC", "PASS", "LAW", "INTRO"]].sum().reset_index()
sum_roles_table = people_df.groupby(["session_id", "role_id"]).people_id.nunique().reset_index()
# Rename columns in sum_table
sum_bills_table.columns = ["session_id"] + [f"{col}_sum" for col in sum_bills_table.columns[1:]]
sum_roles_table.columns = ["session_id", "role_id", "N_lawmakers"]
# Group by people_id and session_id
grouped_table = authors_df_temp.groupby(["people_id", "role_id", "session_id"]).agg(
    INTRO_term=("INTRO", "sum"),
    AIC_term=("AIC", "sum"),
    ABC_term=("ABC", "sum"),
    PASS_term=("PASS", "sum"),
    LAW_term=("LAW", "sum")
).reset_index()

grouped_table = grouped_table.merge(sum_bills_table, on = 'session_id', how = 'left')
grouped_table = grouped_table.merge(sum_roles_table, on = ['session_id', 'role_id'], how = 'left')
# Calculate the ratio by dividing by the corresponding sum
grouped_table["INTRO_term"] /= grouped_table["INTRO_sum"]
grouped_table["AIC_term"] /= grouped_table["AIC_sum"]
grouped_table["ABC_term"] /= grouped_table["ABC_sum"]
grouped_table["PASS_term"] /= grouped_table["PASS_sum"]
grouped_table["LAW_term"] /= grouped_table["LAW_sum"]

grouped_table["LES"] = grouped_table["INTRO_term"]+ grouped_table["AIC_term"] + grouped_table["ABC_term"] + grouped_table["PASS_term"] +grouped_table["LAW_term"] 
grouped_table["LES"] *= grouped_table["N_lawmakers"]/5
grouped_table["LPS"] = grouped_table["INTRO_term"]*grouped_table["N_lawmakers"]

authors_df = authors_df.merge(grouped_table[['session_id', 'people_id', 'LES', 'LPS']], on = ['session_id', 'people_id'], how = 'left')

authors_df = authors_df.merge(matched_authors_bills, on = 'bill_id_new', how = 'left')
authors_df = authors_df.merge(matched_contact_bills, on = 'bill_id_new', how = 'left')
authors_df.loc[authors_df.is_matched_bill != True, 'is_matched_bill'] = False
authors_df.loc[authors_df.is_bill_lobbyist_lawmaker_matched != True, 'is_bill_lobbyist_lawmaker_matched'] = False

authors_df.loc[authors_df.is_bill_lobbyist_lawmaker_matched != True, 'is_bill_lobbyist_lawmaker_matched'] = False
authors_df = authors_df.loc[authors_df.committee_id==0,:]
authors_df.to_csv(machine + 'tables/' + 'authors_table.csv')

lobbyist_client_bill_temp = find_lobbyist_author.loc[find_lobbyist_author.is_bill_lobbyist_lawmaker_matched, ['bill_id_new', 'beneficial_client', 'principal_lobbyist']].drop_duplicates()
sponsors_table_temp_ = contact_table.loc[contact_table.is_sponsors_chamber_1_type_1 == True,:].drop_duplicates(['bill_id_new', 'beneficial_client', 'principal_lobbyist', 'people_id'])

sponsors_clients_lobbyist_contacts = lobbyist_client_bill_temp.merge(sponsors_table_temp_, how = 'left', on = ['bill_id_new', 'beneficial_client', 'principal_lobbyist'])
sponsors_clients_lobbyist_contacts.to_csv(machine + 'tables/' + 'sponsors_clients_lobbyist_contacts_table.csv')



#%%

contact_grouped_bill = contact_table.groupby(['session_id', 'bill_id_new'])

contact_grouped_bill = contact_grouped_bill.agg(
    total_contacts=pd.NamedAgg(column='bill_id_new', aggfunc='count'),
    total_member_contacts = pd.NamedAgg(column='is_member', aggfunc=lambda x: x.dropna().sum()),
    sum_sponsors_chamber_1_type_1=pd.NamedAgg(column='is_sponsors_chamber_1_type_1', aggfunc=lambda x: x.dropna().sum()),
    sum_sponsors_chamber_1_type_2=pd.NamedAgg(column='is_sponsors_chamber_1_type_2', aggfunc=lambda x: x.dropna().sum()),
    sum_sponsors_chamber_2_type_1=pd.NamedAgg(column='is_sponsors_chamber_2_type_1', aggfunc=lambda x: x.dropna().sum()),
    sum_sponsors_chamber_2_type_2=pd.NamedAgg(column='is_sponsors_chamber_2_type_2', aggfunc=lambda x: x.dropna().sum()),
    sum_chair_committee_chamber_1_type_1=pd.NamedAgg(column='is_chair_committee_chamber_1_type_1', aggfunc=lambda x: x.dropna().sum()),
    sum_committee_members_chamber_1_type_1=pd.NamedAgg(column='is_committee_member_chamber_1_type_1', aggfunc=lambda x: x.dropna().sum()),
    sum_committee_members_chamber_1_type_2=pd.NamedAgg(column='is_committee_member_chamber_1_type_2', aggfunc=lambda x: x.dropna().sum()),
    sum_committee_members_chamber_2_type_1=pd.NamedAgg(column='is_committee_member_chamber_2_type_1', aggfunc=lambda x: x.dropna().sum()),
    sum_committee_members_chamber_2_type_2=pd.NamedAgg(column='is_committee_member_chamber_2_type_2', aggfunc=lambda x: x.dropna().sum()),
    sum_committee_members_chamber_1_type_1_excluding_sponsors=pd.NamedAgg(column='is_committee_member_chamber_1_type_1_excluding_sponsors', aggfunc=lambda x: x.dropna().sum()),
    sum_committee_members_chamber_1_type_2_excluding_sponsors=pd.NamedAgg(column='is_committee_member_chamber_1_type_2_excluding_sponsors', aggfunc=lambda x: x.dropna().sum()),
    sum_committee_members_chamber_2_type_1_excluding_sponsors=pd.NamedAgg(column='is_committee_member_chamber_2_type_1_excluding_sponsors', aggfunc=lambda x: x.dropna().sum()),
    sum_committee_members_chamber_2_type_2_excluding_sponsors=pd.NamedAgg(column='is_committee_member_chamber_2_type_2_excluding_sponsors', aggfunc=lambda x: x.dropna().sum()),
    sum_committee_chamber_1_type_1=pd.NamedAgg(column='is_committee_chamber_1_type_1', aggfunc=lambda x: x.dropna().sum()),
    sum_committee_chamber_1_type_2=pd.NamedAgg(column='is_committee_chamber_1_type_2', aggfunc=lambda x: x.dropna().sum()),
    sum_committee_chamber_2_type_1=pd.NamedAgg(column='is_committee_chamber_2_type_1', aggfunc=lambda x: x.dropna().sum()),
    sum_committee_chamber_2_type_2=pd.NamedAgg(column='is_committee_chamber_2_type_2', aggfunc=lambda x: x.dropna().sum()),
    sum_member_chamber_1_excluding_sponsors_committee_members=pd.NamedAgg(column='is_member_chamber_1_excluding_sponsors_committee_members', aggfunc=lambda x: x.dropna().sum()),
    sum_member_chamber_2_excluding_sponsors_committee_members=pd.NamedAgg(column='is_member_chamber_2_excluding_sponsors_committee_members', aggfunc=lambda x: x.dropna().sum()),
    sum_member_chamber_1=pd.NamedAgg(column='is_member_chamber_1', aggfunc=lambda x: x.dropna().sum()),
    sum_member_chamber_2=pd.NamedAgg(column='is_member_chamber_2', aggfunc=lambda x: x.dropna().sum()),
    sum_is_chamber_1=pd.NamedAgg(column='is_chamber_1', aggfunc=lambda x: x.dropna().sum()),
    sum_is_chamber_2=pd.NamedAgg(column='is_chamber_2', aggfunc=lambda x: x.dropna().sum())
    
).reset_index()

committe_bills_df = event_table[['bill_id_new', 'session_id', 'committee_chamber_1_type_1']].drop_duplicates()
committe_bills_df = committe_bills_df.merge(contact_grouped_bill, on = ['session_id', 'bill_id_new'], how = 'left')

committees_authors_df = authors_df.loc[authors_df.sponsor_type_id == 1,:].drop(['people_id', 'name', 'sponsor_order'], axis=1)
# Count the number of party_id == 1 and party_id == 2 for each bill_id_new
party_counts = committees_authors_df.groupby(['bill_id_new', 'party_id']).size().unstack(fill_value=0).reset_index()
party_counts.columns = ['bill_id_new', 'party_1_count', 'party_2_count']
# Count the number of sponsor_type_id == 1 and sponsor_type_id == 2 for each bill_id_new
sponsor_counts = committees_authors_df.groupby(['bill_id_new', 'sponsor_type_id']).size().unstack(fill_value=0).reset_index()
sponsor_counts.columns = ['bill_id_new', 'sponsor_1_count']
# Merge the counts with the original dataframe
committees_authors_df = pd.merge(committees_authors_df, party_counts, on='bill_id_new', how='left')
committees_authors_df = pd.merge(committees_authors_df, sponsor_counts, on='bill_id_new', how='left')

# Drop the specified columns
committees_authors_df = committees_authors_df.drop(columns=['party_id', 'role_id', 'sponsor_type_id', 'committee_sponsor'])

# Drop duplicates by bill_id_new
committees_authors_df = committees_authors_df.drop_duplicates(subset='bill_id_new')

# Assuming your DataFrame is named event_table
# First, filter the DataFrame to get the relevant rows
# Assuming 'df' is your dataframe
df = event_table.sort_values('date')  # Ensure the data is sorted by date

# Filter rows where event = 9 and committee_id_event = committee_chamber_1_type_1
filtered_df = df[(df['event'] == 9) & (df['committee_id_event'] == df['committee_chamber_1_type_1'])]

# Group by bill_id_new and get the first date for each group
first_dates = filtered_df.groupby('bill_id_new')['date'].first()

# Initialize an empty dictionary to store the results
results = {}

i=0
# For each bill_id_new, find whether there is any date AFTER the first date
for bill_id_new, first_date in first_dates.iteritems():
    if i%100==0:
        print(i)
    later_dates = df[(df['bill_id_new'] == bill_id_new) & (df['date'] > first_date)]
    if not later_dates.empty:
        # If there is any date AFTER the first date, save the value of that event
        results[bill_id_new] = (later_dates['event'].values[0], later_dates['committee_id_event'].values[0])
    i+=1
    
# Convert the results to a dataframe
results_df = pd.DataFrame.from_dict(results, orient='index', columns=['event_after_first_referral', 'committee_after_first_referral']).reset_index()
results_df.columns = ['bill_id_new', 'event_after_first_referral', 'committee_after_first_referral']
committees_authors_df = committees_authors_df.merge(results_df.reset_index(), how = 'left', on = 'bill_id_new')

committees_authors_df['passes_first_referral'] = (committees_authors_df['event_after_first_referral'] != 1) & ~((committees_authors_df['event_after_first_referral'] == 9) & pd.isna(committees_authors_df['committee_after_first_referral'])) & (~pd.isna(committees_authors_df['event_after_first_referral']))

committee_members_df = pd.read_csv(machine + legi_subfolder + 'committees_members_all.csv')
committee_count = committee_members_df.groupby(['session_id','committee_id']).people_id_inferred.count().reset_index()
committee_count.columns = ['session_id', 'committee_id', 'committee_size']
committees_authors_df.pop('committee_id')
committees_authors_df = committees_authors_df.merge(committee_count, how='left', left_on = ['session_id', 'committee_chamber_1_type_1'], right_on = ['session_id', 'committee_id'])

committe_bills_df = committe_bills_df.merge(committees_authors_df.drop('committee_chamber_1_type_1', axis=1), on = ['session_id', 'bill_id_new'], how = 'left')
committe_bills_df.to_csv(machine + 'tables/' + 'committees_table.csv')


#%%

parties_sub = parties_df.loc[~pd.isna(parties_df.people_id),:].drop_duplicates(['people_id', 'year', 'filing_bimonth', 'principal_lobbyist', 'beneficial_client'])

intensity_lobbying_cont = parties_sub.groupby(['people_id', 'session_id']).parties_lobbied_id.count().reset_index()
intensity_lobbying_cont.columns = ['people_id', 'session_id', 'intensity_contacts']
intensity_lobbying_comp = parties_sub.groupby(['people_id', 'session_id']).total_compensation.sum().reset_index()
intensity_lobbying_comp.columns = ['people_id', 'session_id', 'intensity_expenditures']

intensity_lobbying = intensity_lobbying_cont.merge(intensity_lobbying_comp, on = ['people_id', 'session_id'])


intensity_lobbying.to_csv(machine + 'tables/' + 'intensity_lobbying.csv')

parties_sub = parties_df.loc[parties_df.is_matched_bill == False & ~pd.isna(parties_df.people_id),:].drop_duplicates(['people_id', 'year', 'filing_bimonth', 'principal_lobbyist', 'beneficial_client'])
intensity_lobbying_cont = parties_sub.groupby(['people_id', 'session_id']).parties_lobbied_id.count().reset_index()
intensity_lobbying_cont.columns = ['people_id', 'session_id', 'intensity_contacts']
intensity_lobbying_comp = parties_sub.groupby(['people_id', 'session_id']).total_compensation.sum().reset_index()
intensity_lobbying_comp.columns = ['people_id', 'session_id', 'intensity_expenditures']

intensity_lobbying = intensity_lobbying_cont.merge(intensity_lobbying_comp, on = ['people_id', 'session_id'])


parties_sub = parties_df.loc[parties_df.is_matched_bill == False & ~pd.isna(parties_df.people_id),:].drop_duplicates(['people_id', 'year', 'filing_bimonth', 'principal_lobbyist', 'beneficial_client'])
intensity_lobbying_cont = parties_sub.groupby(['people_id', 'year']).parties_lobbied_id.count().reset_index()
intensity_lobbying_cont.columns = ['people_id', 'year', 'intensity_contacts']
intensity_lobbying_comp = parties_sub.groupby(['people_id', 'year']).total_compensation.sum().reset_index()
intensity_lobbying_comp.columns = ['people_id', 'year','intensity_expenditures']

intensity_lobbying = intensity_lobbying_cont.merge(intensity_lobbying_comp, on = ['people_id', 'year'])
intensity_lobbying.to_csv(machine + 'tables/' + 'intensity_lobbying_before_year_sem.csv')

parties_sub = contact_table.loc[contact_table.is_sponsors_chamber_1_type_1,:].drop_duplicates(['people_id', 'year', 'filing_bimonth', 'principal_lobbyist', 'beneficial_client'])

intensity_lobbying_cont = parties_sub.groupby(['people_id', 'session_id']).parties_lobbied_id.count().reset_index()
intensity_lobbying_cont.columns = ['people_id', 'session_id', 'intensity_contacts']
intensity_lobbying_comp = parties_sub.groupby(['people_id', 'session_id']).total_compensation.sum().reset_index()
intensity_lobbying_comp.columns = ['people_id', 'session_id', 'intensity_expenditures']

intensity_lobbying = intensity_lobbying_cont.merge(intensity_lobbying_comp, on = ['people_id', 'session_id'])

intensity_lobbying.to_csv(machine + 'tables/' + 'intensity_lobbying_sponsors.csv')

#%%
assembly_expenses = pd.read_csv(machine + expenditures_subfolder + 'expenditures_assembly_members.csv')
assembly_expenses = assembly_expenses.loc[~pd.isna(assembly_expenses.start_service_date),:]

assembly_expenses['start_month_year'] = assembly_expenses.start_year + 1/12 * assembly_expenses.start_month
assembly_expenses['end_month_year'] = assembly_expenses.end_year + 1/12 * assembly_expenses.end_month
assembly_expenses['length_service'] = (assembly_expenses['end_month_year']-assembly_expenses['start_month_year'])*12+1
assembly_expenses['monthly_amt']= assembly_expenses['amt']/assembly_expenses['length_service']
assembly_expenses['expense_MarSep'] = False
assembly_expenses['expense_MarSep'] = assembly_expenses['start_year'] == assembly_expenses['end_year']

payee_earliest_start = assembly_expenses.groupby('PAYEE').start_month_year.min().reset_index()
payee_earliest_start.columns = ['PAYEE', 'payee_earliest_start']
assembly_expenses = assembly_expenses.merge(payee_earliest_start, on = 'PAYEE', how = 'left')
assembly_expenses['payee_experience'] = assembly_expenses.start_month_year - assembly_expenses.payee_earliest_start

assembly_expenses_new = assembly_expenses.copy()
bimonth_check_df = pd.DataFrame({'bimonth_check': range(1, 7)})

# Merge assembly_expenses_new with bimonth_check_df
assembly_expenses_new = pd.merge(assembly_expenses_new, bimonth_check_df, how='cross')
assembly_expenses_new['match_bimonth'] = (assembly_expenses_new['start_month'] >= ((assembly_expenses_new['bimonth_check'] - 1) * 2 + 1)) & (assembly_expenses_new['end_month'] < (2 * assembly_expenses_new['bimonth_check']))
assembly_expenses_new = assembly_expenses_new.loc[assembly_expenses_new.match_bimonth,:]

assembly_expenses_new['filing_semester'] = assembly_expenses_new['bimonth_check'].apply(lambda x: round(x/6)+1)

people_df = pd.read_csv(machine + legi_subfolder +'people_all.csv')
sessions_df = pd.read_csv(machine + legi_subfolder +'sessions_all.csv')

people_df = people_df.merge(sessions_df, on = 'session_id', how='left')
people_earliest_start = people_df.groupby('people_id').session_odd_year.min().reset_index()
people_earliest_start.columns = ['people_id', 'earliest_start_people']
people_role_earliest_start = people_df.groupby(['people_id', 'role_id']).session_odd_year.min().reset_index()
people_role_earliest_start.columns = ['people_id', 'role_id', 'earliest_start_people_role']
people_df = people_df.merge(people_earliest_start, on = 'people_id', how = 'left')
people_df = people_df.merge(people_role_earliest_start, on = ['people_id', 'role_id'], how = 'left')
people_df['terms'] = (people_df.session_odd_year - people_df.earliest_start_people)/2
people_df['terms_in_role'] = (people_df.session_odd_year - people_df.earliest_start_people_role)/2

assembly_expenses_new = assembly_expenses_new.merge(people_df[['people_id', 'session_id', 'terms', 'terms_in_role']], on = ['people_id', 'session_id'], how='left')
def legislative_aggregation(group):
    legislative_rows = group['DESCRIPTION'].str.contains("LEGISLATIVE")
    return pd.Series({
        'legislative_amt': group.loc[legislative_rows, 'monthly_amt'].sum(),
        'legislative_experience': group.loc[legislative_rows, 'payee_experience'].sum(),
        'distinct_legislative_units': group.loc[legislative_rows, 'PAYEE'].nunique()
    })

# Group by (people_id, session_id) and calculate the desired me§trics

#%%


senate_expenses = pd.read_csv(machine + expenditures_subfolder + 'expenditures_senate_members.csv')
senate_expenses = senate_expenses.loc[~pd.isna(senate_expenses.start_service_date),:]

senate_expenses['start_month_year'] = senate_expenses.start_year + 1/12 * senate_expenses.start_month
senate_expenses['end_month_year'] = senate_expenses.end_year + 1/12 * senate_expenses.end_month
senate_expenses['length_service'] = (senate_expenses['end_month_year']-senate_expenses['start_month_year'])*12+1
senate_expenses['monthly_amt']= senate_expenses['amt']/senate_expenses['length_service']
senate_expenses['expense_MarSep'] = False
senate_expenses['expense_MarSep'] = senate_expenses['start_year'] == senate_expenses['end_year']

payee_earliest_start = senate_expenses.groupby('PAYEE').start_month_year.min().reset_index()
payee_earliest_start.columns = ['PAYEE', 'payee_earliest_start']
senate_expenses = senate_expenses.merge(payee_earliest_start, on = 'PAYEE', how = 'left')
senate_expenses['payee_experience'] = senate_expenses.start_month_year - senate_expenses.payee_earliest_start

senate_expenses_new = senate_expenses.copy()
bimonth_check_df = pd.DataFrame({'bimonth_check': range(1, 7)})

# Merge senate_expenses_new with bimonth_check_df
senate_expenses_new = pd.merge(senate_expenses_new, bimonth_check_df, how='cross')
senate_expenses_new['match_bimonth'] = (senate_expenses_new['start_month'] >= ((senate_expenses_new['bimonth_check'] - 1) * 2 + 1)) & (senate_expenses_new['end_month'] < (2 * senate_expenses_new['bimonth_check']))
senate_expenses_new = senate_expenses_new.loc[senate_expenses_new.match_bimonth,:]

senate_expenses_new['filing_semester'] = senate_expenses_new['bimonth_check'].apply(lambda x: round(x/6)+1)

people_df = pd.read_csv(machine + legi_subfolder +'people_all.csv')
sessions_df = pd.read_csv(machine + legi_subfolder +'sessions_all.csv')

people_df = people_df.merge(sessions_df, on = 'session_id', how='left')
people_earliest_start = people_df.groupby('people_id').session_odd_year.min().reset_index()
people_earliest_start.columns = ['people_id', 'earliest_start_people']
people_role_earliest_start = people_df.groupby(['people_id', 'role_id']).session_odd_year.min().reset_index()
people_role_earliest_start.columns = ['people_id', 'role_id', 'earliest_start_people_role']
people_df = people_df.merge(people_earliest_start, on = 'people_id', how = 'left')
people_df = people_df.merge(people_role_earliest_start, on = ['people_id', 'role_id'], how = 'left')
people_df['terms'] = (people_df.session_odd_year - people_df.earliest_start_people)/2
people_df['terms_in_role'] = (people_df.session_odd_year - people_df.earliest_start_people_role)/2

senate_expenses_new = senate_expenses_new.merge(people_df[['people_id', 'session_id', 'terms', 'terms_in_role']], on = ['people_id', 'session_id'], how='left')
def legislative_aggregation(group):
    legislative_rows = group['DESCRIPTION'].str.contains("LEGISLATIVE")
    return pd.Series({
        'legislative_amt': group.loc[legislative_rows, 'monthly_amt'].sum(),
        'legislative_experience': group.loc[legislative_rows, 'payee_experience'].sum(),
        'distinct_legislative_units': group.loc[legislative_rows, 'PAYEE'].nunique()
    })

# Group by (people_id, session_id) and calculate the desired metrics
#%%

result_df = assembly_expenses_new.groupby(['people_id', 'session_id']).apply(legislative_aggregation).reset_index()

# Add sum_amt and distinct_units columns
result_df[['sum_amt', 'sum_experience', 'distinct_units']] = assembly_expenses.groupby(['people_id', 'session_id']).agg(
    sum_amt=('amt', 'sum'),
    sum_experience = ('payee_experience', 'sum'),
    distinct_units=('PAYEE', 'nunique')
).reset_index()[['sum_amt', 'sum_experience', 'distinct_units']]

result_df = result_df.merge(people_df[['people_id', 'session_id' ,'terms', 'terms_in_role']], on = ['people_id', 'session_id'], how='left')

result_df.to_csv(machine + 'tables/' + 'assembly_members_aggregated_expenses.csv')

#%%


result_df = senate_expenses_new.groupby(['people_id', 'session_id']).apply(legislative_aggregation).reset_index()

# Add sum_amt and distinct_units columns
result_df[['sum_amt', 'sum_experience', 'distinct_units']] = assembly_expenses.groupby(['people_id', 'session_id']).agg(
    sum_amt=('amt', 'sum'),
    sum_experience = ('payee_experience', 'sum'),
    distinct_units=('PAYEE', 'nunique')
).reset_index()[['sum_amt', 'sum_experience', 'distinct_units']]

result_df = result_df.merge(people_df[['people_id', 'session_id' ,'terms', 'terms_in_role']], on = ['people_id', 'session_id'], how='left')

result_df.to_csv(machine + 'tables/' + 'senate_members_aggregated_expenses.csv')
# Display the result DataFrame
print(result_df)
#%%
result_df = assembly_expenses.groupby(['people_id', 'session_id']).apply(legislative_aggregation).reset_index()

# Add sum_amt and distinct_units columns
result_df[['sum_amt', 'sum_experience', 'distinct_units']] = assembly_expenses.groupby(['people_id', 'session_id']).agg(
    sum_amt=('amt', 'sum'),
    sum_experience = ('payee_experience', 'sum'),
    distinct_units=('PAYEE', 'nunique')
).reset_index()[['sum_amt', 'sum_experience', 'distinct_units']]

result_df = result_df.merge(people_df[['people_id', 'session_id', 'terms', 'terms_in_role']], on = ['people_id', 'session_id'], how='left')

result_df.to_csv(machine + 'tables/' + 'assembly_members_aggregated_expenses.csv')
# Display the result DataFrame
print(result_df)

#%%

assembly_expenses = pd.read_csv(machine + 'tables/' + 'assembly_members_aggregated_expenses.csv')

senate_expenses = pd.read_csv(machine + 'tables/' + 'senate_members_aggregated_expenses.csv')

expenses = pd.concat((assembly_expenses, senate_expenses), axis=0)

expenses.to_csv(machine + 'tables/' + 'members_aggregated_expenses.csv')

#%%
fig, ax = plt.subplots()
ax.hist(individual_lobbyists_dataset.groupby('PrincipalLobbyist').TotalIndividualLobbyists.mean(), bins = 40)
ax.set_xlabel('Total number of individual lobbyists')
ax.set_ylabel('Frequency')
fig.savefig('individual_lobbysts.png', dpi=300)

#%%


issueComp = pd.DataFrame(subjects_lobbied.groupby('MainIssue').TotalCompensation.sum()/4/1e6).reset_index()
length_split = math.ceil(len(issueComp)/2)
issueComp.MainIssue = issueComp.MainIssue.str.replace('&', 'and')
issueComp = issueComp.rename(columns = {'TotalCompensation': 'TotalCompensation (1,000,000)'})
issueComp_split = pd.concat([issueComp.iloc[:length_split-1,:].reset_index(drop=True), issueComp.iloc[length_split:,:].reset_index(drop=True)],axis=1,ignore_index=False)
print(issueComp_split.style.format(formatter="{0:.2f}", subset=['TotalCompensation (1,000,000)']).to_latex(multicol_align = "|c|",sparse_columns = True, hrules=True))

#%%
groupLobbyist = lobbyists_dataset.groupby(['PrincipalLobbyist'])  
groupLobbyistRetained = lobbyists_dataset[lobbyists_dataset.RetainedFlag==True].groupby(['PrincipalLobbyist'])  
groupLobbyistEmployed = lobbyists_dataset[lobbyists_dataset.RetainedFlag==False].groupby(['PrincipalLobbyist'])  
groupLobbyistYear = lobbyists_dataset.groupby(['PrincipalLobbyist', 'Year'])
groupLobbyistYearRetained = lobbyists_dataset[lobbyists_dataset.RetainedFlag==True].groupby(['PrincipalLobbyist', 'Year'])  
groupLobbyistYearEmployed = lobbyists_dataset[lobbyists_dataset.RetainedFlag==False].groupby(['PrincipalLobbyist', 'Year'])  
groupYear = lobbyists_dataset.groupby(['Year'])
groupYearRetained = lobbyists_dataset[lobbyists_dataset.RetainedFlag==True].groupby(['Year'])  
groupLYearEmployed = lobbyists_dataset[lobbyists_dataset.RetainedFlag==False].groupby(['Year']) 

cols = [('Mean', 'external'), ('Mean', 'in-house'), ('SD', 'external'), ('SD', 'in-house'),
          ('Min', 'external'), ('Min', 'in-house'), ('Max', 'external'), ('Max', 'in-house'),
          ('Share zeros', 'external'), ('Share zeros', 'in-house')]
lobbyist_table = pd.DataFrame(index=['Lobbying firms per year', 'Clients', 'Compensation per year (1000s)', '\\quad Compensation', '\\quad Compensation per client', '\\quad Reimburses'], columns = pd.MultiIndex.from_tuples(cols))
#lobbyist_table = pd.DataFrame(index=['Lobbying firms per year', 'Clients', 'Compensation per year (1000s)', '\\quad Compensation', '\\quad Compensation per client', '\\quad Reimburses', '\\quad Other', '\\quad Coalition contr.'], columns = pd.MultiIndex.from_tuples(cols))

#lobbyist_table = pd.DataFrame(index=['Clients', '\\quad \\% Retained', 'Expenses', '\\quad Compensation', '\\quad Reimburses', '\\quad Non-lobbying expenses', '\\quad Coalition contribution'], columns = pd.MultiIndex.from_tuples(cols))

#print(lobbyist_table.style.to_latex(multicol_align = "|c|", sparse_columns =True, hrules=True))

fig, ax = plt.subplots()
ax.hist(np.log10(groupLobbyistYearRetained.Compensation.sum().reset_index().groupby('PrincipalLobbyist').Compensation.mean()+1), alpha=0.5, bins=50)
ax.hist(np.log10(groupLobbyistYearEmployed.Compensation.sum().reset_index().groupby('PrincipalLobbyist').Compensation.mean()+1), alpha=0.5, bins=50)
ax.set_xlabel('Log of mean annual compensation')
ax.set_ylabel('Frequency')
fig.savefig('annual_compensation.png', dpi=300)


#%%

lobbyist_table.iloc[0,np.arange(0,len(cols),2)] = returnBasicStats(groupYearRetained.PrincipalLobbyist.nunique(), include_share_zeros=True)
lobbyist_table.iloc[0,np.arange(1,len(cols),2)] = returnBasicStats(groupLYearEmployed.PrincipalLobbyist.nunique(), include_share_zeros=True)


lobbyist_table.iloc[1,np.arange(0,len(cols),2)] = returnBasicStats(groupLobbyistYearRetained.BeneficialClient.count().reset_index().groupby('PrincipalLobbyist').BeneficialClient.mean(), include_share_zeros=True)
lobbyist_table.iloc[1,np.arange(1,len(cols),2)] = returnBasicStats(groupLobbyistYearEmployed.BeneficialClient.count().reset_index().groupby('PrincipalLobbyist').BeneficialClient.mean(), include_share_zeros=True)

lobbyist_table.iloc[2, :] = np.nan

lobbyist_table.iloc[3,np.arange(0,len(cols),2)] = returnBasicStats(groupLobbyistYearRetained.Compensation.sum().reset_index().groupby('PrincipalLobbyist').Compensation.mean(), include_share_zeros=True)
lobbyist_table.iloc[3,np.arange(1,len(cols),2)] = returnBasicStats(groupLobbyistYearEmployed.Compensation.sum().reset_index().groupby('PrincipalLobbyist').Compensation.mean(), include_share_zeros=True)

lobbyist_table.iloc[4,np.arange(0,len(cols),2)] = returnBasicStats(groupLobbyistYearRetained.Compensation.mean().reset_index().groupby('PrincipalLobbyist').Compensation.mean(), include_share_zeros=True)
lobbyist_table.iloc[4,np.arange(1,len(cols),2)] = returnBasicStats(groupLobbyistYearEmployed.Compensation.mean().reset_index().groupby('PrincipalLobbyist').Compensation.mean(), include_share_zeros=True)

lobbyist_table.iloc[5,np.arange(0,len(cols),2)] = returnBasicStats(groupLobbyistYearRetained.ReimbursedExpenses.sum().reset_index().groupby('PrincipalLobbyist').ReimbursedExpenses.mean(), include_share_zeros=True)
lobbyist_table.iloc[5,np.arange(1,len(cols),2)] = returnBasicStats(groupLobbyistYearEmployed.ReimbursedExpenses.sum().reset_index().groupby('PrincipalLobbyist').ReimbursedExpenses.mean(), include_share_zeros=True)

#lobbyist_table.iloc[6,np.arange(0,len(cols),2)] = returnBasicStats(groupLobbyistYearRetained.NonLobbyingExpenses.sum().reset_index().groupby('PrincipalLobbyist').NonLobbyingExpenses.mean(), include_share_zeros=True)
#lobbyist_table.iloc[6,np.arange(1,len(cols),2)] = returnBasicStats(groupLobbyistYearEmployed.NonLobbyingExpenses.sum().reset_index().groupby('PrincipalLobbyist').NonLobbyingExpenses.mean(), include_share_zeros=True)

#lobbyist_table.iloc[6,np.arange(0,len(cols),2)] = returnBasicStats(groupLobbyistYearRetained.CoalitionContribution.sum().reset_index().groupby('PrincipalLobbyist').CoalitionContribution.mean(), include_share_zeros=True)
#lobbyist_table.iloc[6,np.arange(1,len(cols),2)] = returnBasicStats(groupLobbyistYearEmployed.CoalitionContribution.sum().reset_index().groupby('PrincipalLobbyist').CoalitionContribution.mean(), include_share_zeros=True)

lobbyist_table.iloc[3:,:-1] = lobbyist_table.iloc[3:,:-1]/1000

lobbyists_style = lobbyist_table.style.format(formatter="{0:.2f}", na_rep='')
lobbyists_style = lobbyists_style.format(formatter="{:.1f}", subset=pd.IndexSlice[['Clients'], :])
lobbyists_style = lobbyists_style.format(formatter="{0:.0f}", subset=pd.IndexSlice[['Lobbying firms per year'], :])
#lobbyists_style = lobbyists_style.format(formatter="{:.2f}", subset=pd.IndexSlice[['\\quad \\% Retained'], :])

print(lobbyists_style.to_latex(multicol_align = "|c|", sparse_columns =True, hrules=True))

#%%

activities = [state_activities, municipal_activities]

cond = (activities[i].FilingType=='BI-MONTHLY') | (activities[i].FilingType=='BI-MONTHLY AMENDMENT')
groupActivity = activities[i][cond].groupby(['Activity'])
groupActivityIssue = activities[i][cond].groupby(['Activity', 'Issue'])
groupActivityClient = activities[i][cond].groupby(['Activity', 'BeneficialClient'])
groupActivityLobbyist = activities[i][cond].groupby(['Activity', 'PrincipalLobbyist'])
groupActivityCommunication = activities[i][cond].groupby(['Activity', 'TypeOfLobbyingCommunication'])

# count issues
ci = groupActivity.Issue.count()
# client per issue
cpi = groupActivityIssue.BeneficialClient.nunique().reset_index().groupby('Activity').BeneficialClient.mean()
# issues per client
ipc = groupActivityClient.Issue.nunique().reset_index().groupby('Activity').Issue.mean()
# lobbyists per issue
lpi = groupActivityIssue.PrincipalLobbyist.nunique().reset_index().groupby('Activity').PrincipalLobbyist.mean()
# issues per lobbyist
ipl = groupActivityLobbyist.Issue.nunique().reset_index().groupby('Activity').Issue.mean()
# coomunication share
communication_share = groupActivityCommunication.TypeOfLobbyingCommunication.count().unstack()
communication_share = communication_share.div(communication_share.sum(1), axis=0)
communication_share.pop('GRASSROOTS')

activity_table = pd.concat([ci,cpi,ipc,lpi,ipl,communication_share], axis=1)
activity_table.index = activity_table.index.str.title()

cols = ['Bi-monthly & reports', 'Clients & per issue', 'Issues & per client',
                          'Lobbyists & per issue', 'Issues & per lobbyist', '\\% direct and & grassroot',
                          '\\% direct & lobbying', '\\% monitoring & only']
activity_table.columns = splitColumnHeaders(cols)
activity_style = activity_table.style.format(formatter="{0:.2f}", na_rep=0.00)
activity_style = activity_style.format(formatter="{0:.0f}", subset = [activity_table.columns[0]], na_rep='')

print(activity_style.to_latex(hrules=True))

#%%

cond = (parties_lobbied.FilingType=='BI-MONTHLY') | (parties_lobbied.FilingType=='BI-MONTHLY AMENDMENT')
groupBody = parties_lobbied[cond].groupby('GovernmentBody/Agency')
groupBodyClient = parties_lobbied[cond].groupby(['GovernmentBody/Agency', 'BeneficialClient'])
groupBodyLobbyist = parties_lobbied[cond].groupby(['GovernmentBody/Agency', 'PrincipalLobbyist'])
groupBodyClientIssue = parties_lobbied[cond].groupby(['GovernmentBody/Agency', 'BeneficialClient', 'LobbyingFocusNumberOrDescription'])

cp = groupBody.PartyName.count()
ppc_mean = groupBodyClient.PartyName.nunique().reset_index().groupby('GovernmentBody/Agency').mean()
ppc_std = groupBodyClient.PartyName.nunique().reset_index().groupby('GovernmentBody/Agency').std()
ppl_mean = groupBodyLobbyist.PartyName.nunique().reset_index().groupby('GovernmentBody/Agency').mean()
ppl_std = groupBodyLobbyist.PartyName.nunique().reset_index().groupby('GovernmentBody/Agency').std()
ppci_mean = groupBodyClientIssue.PartyName.nunique().reset_index().groupby('GovernmentBody/Agency').mean()
ppci_std = groupBodyClientIssue.PartyName.nunique().reset_index().groupby('GovernmentBody/Agency').std()

parties_table = pd.concat([cp, ppc_mean, ppc_std, ppl_mean, ppl_std, ppci_mean, ppci_std], axis=1)
parties_table.index = parties_table.index.str.title()
cols = [('Obs',''), ('Parties per client', 'Mean'), ('Parties per client', 'SD'), ('Parties per lobbyist', 'Mean'),
        ('Parties per lobbyist', 'SD'), ('Parties per client-issue', 'Mean'), ('Parties per client-issue', 'SD')]
parties_table.columns = pd.MultiIndex.from_tuples(cols)
parties_style = parties_table.style.format(formatter="{0:.2f}", na_rep=0.00)
parties_style = parties_style.format(formatter="{0:.0f}", subset = [parties_style.columns[0]], na_rep='')

print(parties_style.to_latex(multicol_align = "|c|", sparse_columns =True, hrules=True))

#%%

cond = (parties_lobbied.FilingType=='BI-MONTHLY') | (parties_lobbied.FilingType=='BI-MONTHLY AMENDMENT')
cond = cond & ((parties_lobbied['GovernmentBody/Agency'] == 'SENATE'))# | (parties_lobbied['GovernmentBody/Agency'] == 'SENATE COMMITTEE'))
cond = cond & ((parties_lobbied['TypeOfLobbyingCommunication'] == 'DIRECT LOBBYING') | (parties_lobbied['TypeOfLobbyingCommunication'] == 'BOTH DIRECT AND GRASSROOTS'))
cond = cond & (parties_lobbied.LobbyingFocus == 'LEGISLATIVE BILL')
parties_df = parties_lobbied[cond]
