#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 19:07:22 2023

@author: michelev
"""

import numpy as np
import pandas as pd

import re

#import spacy
#import en_core_web_sm
from nltk.corpus import words as nltk_words
from nltk.corpus import stopwords

#import enchant

from unidecode import unidecode

from nameparser import HumanName
from nameparser.config import CONSTANTS
CONSTANTS.titles.add('assembly member')
CONSTANTS.titles.add('assemblywoman')
CONSTANTS.titles.add('assemblyman')

from fuzzywuzzy import fuzz, process
from jarowinkler import jarowinkler_similarity
from textdistance import levenshtein, hamming

#%%

def returnBasicStats(groupObject, include_share_zeros = False):
    
    stats = [groupObject.mean(), groupObject.std(), groupObject.min(), groupObject.max()]
    
    if include_share_zeros:
        
        stats = stats + [sum(groupObject == 0)/len(groupObject)]
    
    return stats

def splitSubjectsLobbied(s):

    if pd.isna(s) == False:

        return s.split('\n')[0].split('-')[0].split('–')[0].rstrip()
    
    else:
        
        return s

def extractCompensation(s):

    if pd.isna(s) == False:

        return int(s.replace('$', '').replace(',', ''))
    
    else:
        
        
        return s

def remove_accents_from_string(s):
    if isinstance(s, str):  # Check if the value is a string
        return unidecode(s)
    else:
        return s
    
def extract_years(row):
    if isinstance(row, str): 
        years = [int(y) for y in row.split() if y.isdigit()]
    else:
        years = [row]
    return years

def extract_city_state_phone(col):
    
    col = col.split(' ')
    no_entries = len(col)
    phone = col[no_entries-1]
    state = col[no_entries-2]
    city = ' '.join([city_word.capitalize() for city_word in col[:no_entries-2]])
    
    return city, state, phone

def renameColumn(col):
    
    if col == 'GovernmentBody/Agency':
        
        new_col = 'govt_body'
        
    elif col == 'LobbyingFocus':
    
        new_col = 'focus_level'
        
    elif col == 'LobbyingFocusNumberOrDescription':
    
        new_col = 'focus'
        
    elif col == 'TypeOfLobbyingCommunication':
    
        new_col = 'communication'
        
    elif col == 'TypeOfLobbyist':
        
        new_col = 'lobbyist_type'
        
    else:
        
        new_col = ''
        
        for char in col:
            if char.isupper():
                new_col += '_' + char.lower()
            else:
                new_col += char.lower()
        
        
        
        # Remove leading underscore if present
        new_col = new_col.lstrip('_')
        
    
    return new_col
    


def splitColumnHeaders(column_list):
    
    transformed_column_list = []
    
    for col in column_list:
        
        split = col.split('&')
        if len(split)==1:
            transformed_column_list.append(col)
        else:
            transformed_column_list.append( "\makecell[b]{" + split[0] + " \\\ " + split[1]  + "}")
            
    return transformed_column_list

def stripOnlyString(string):
    
    new_string = ''
    
    if pd.isna(string) == False:
        
        new_string = string.strip()
    
    else:
        
        new_string = string
        
    return new_string

def fromDateToBiMonth(date):
    
    
def fromFilingPeriodToBimonth(filing_period):
    
    dict_bimonth = {'JANUARY/FEBRUARY': 1,
                    'MARCH/APRIL': 2,
                    'MAY/JUNE': 3,
                    'JULY/AUGUST': 4,
                    'SEPTEMBER/OCTOBER': 5,
                    'NOVEMBER/DECEMBER': 6}
    
    return dict_bimonth[filing_period]
 

# Custom scoring function
def custom_scorer(query, choice):
    
    less_important_words = ['NY', 'NYS', 'NEW', 'YORK']
    # Split the query and choice into words
    query_words = query.split()
    choice_words = choice.split()
    
    # Initialize a score
    score = 0
    
    # Calculate the score for each word pair
    for query_word in query_words:
        for choice_word in choice_words:
            # Assign a weight based on importance
            weight = 1 if query_word not in less_important_words else 0.5
            
            # Add the weighted score to the total score
            score += fuzz.ratio(query_word, choice_word) * weight
    
    return score

def fromYearToSessionId(year, sessions_df):
     
     if isinstance(year, str):
         pattern = r'\b\d{4}\b'
         years = re.findall(pattern, year)
         years = [int(year) for year in years]
         return sessions_df.loc[(sessions_df.session_odd_year == years[0]) | (sessions_df.session_even_year == years[0]),'session_id'].values[0]
     
     return sessions_df.loc[(sessions_df.session_odd_year == year) | (sessions_df.session_even_year == year),'session_id'].values[0]

def fuzzy_match(row, choices, column_to_match, scorer=fuzz.ratio, threshold=70):
    """
    Function to perform fuzzy matching using fuzzywuzzy.
    
    Args:
    row: A row from the first dataframe.
    choices: A list of strings from the second dataframe column you want to match against.
    scorer: The scoring function to use (default is fuzz.ratio).
    threshold: The minimum score for a match to be considered valid (default is 90).
    
    Returns:
    The best match from the choices if it meets the threshold, otherwise None.
    """
    best_match, score, index = process.extractOne(row[column_to_match], choices, scorer=scorer)
    
    return index if score >= threshold else None
def findBillCode(focus_string):
    
    pattern = r'(S*\d+)|(S\s*\d+)|(S\.\s*\d+)|(S\.\s*\d+)|(A*\d+)|(A\s*\d+)|(A\.\s*\d+)|(A\.\s*\d+)'
    matches = re.findall(pattern, focus_string)
    
    special_substrings = []
    for match in matches:
        for group in match:
            if group:
                special_substrings.append(group)
    if len(special_substrings)>0:
    
        new_focus_bill = special_substrings[0].replace(' ', '').replace('.', '');
       
        prefix = new_focus_bill[0]
        numbers = new_focus_bill[1:]
        
        zeros_to_add = 6 - len(new_focus_bill)
        if zeros_to_add > 0:
            adjusted_string = prefix + '0' * zeros_to_add + numbers
            new_focus_bill = adjusted_string
    
        return new_focus_bill
    
    return focus_string

def returnFullName(name_dict, substitute_nickname = False, drop_middle_name = False, new_nicknames = None):
    
    full_name = ''
    
    for col in ['first_name', 'middle_name', 'last_name', 'suffix']:
        
        if pd.isna(name_dict[col]) == False:
            
            if (col == 'first_name') & (((pd.isna(name_dict['nickname']) == False) & (pd.isna(name_dict['first_name']) == True)) | ((pd.isna(name_dict['nickname']) == False) & (substitute_nickname == True))):
                col = 'nickname'
            if (col == 'first_name') & (substitute_nickname == True) & (new_nicknames != None):
                if name_dict['first_name'] in new_nicknames.keys():
                    full_name = full_name + str(new_nicknames[name_dict['first_name']]) + ' '
                    continue
            if (col == 'middle_name') & (drop_middle_name == True):
                continue
            else:
                full_name = full_name + str(name_dict[col]) + ' '
        
    return full_name.rstrip()

def manipulatePartyName(identifier, df, dictionary, drop_words, substitute_manual, new_nicknames, committee_df_dict, people_df_dict, homonim_names_dict, similar_names_dict):
    
    name = df.loc[df.parties_lobbied_id == identifier, "party_name"].values[0]
    
    
            
    chamber_id = df.loc[df.parties_lobbied_id == identifier, "chamber_id"].values[0]
    session_id = df.loc[df.parties_lobbied_id == identifier, "session_id"].values[0]
    role_id = 0
    
    if chamber_id == 73:
        
        committee_df = committee_df_dict[session_id].loc[committee_df_dict[session_id].chamber_name == 'ASSEMBLY', :]
        
    elif chamber_id == 74:
        
        committee_df = committee_df_dict[session_id].loc[committee_df_dict[session_id].chamber_name == 'SENATE', :]
    
    if chamber_id == 73:
        
        role_id = 1
        
    elif chamber_id == 74:
        
        role_id = 2
        
    session_ids = list(people_df_dict.keys())
    
    name_dict = {'contacted_staff_counsel': False,
                 'is_staff_counsel': False,
                 'entire_body': False,
                 'majority_body': False,
                 'minority_body': False,
                 'entire_committee': False,
                 'committee': None,
                 'other': False,
                 'people_id': None,
                 'check_rule': ''}

    
    original_name = name
    
    name = name.upper()
    name = name.replace('.', '. ').replace('  ', ' ')
    name = name.replace(",", "")
    name = name.replace(".", "")
    name = name.replace("(", "")
    name = name.replace(")", "")
    name = name.replace("@", "")
    
    for substitutes_manual in substitute_manual.keys():
        
        if name in substitutes_manual:
            
            role = substitute_manual[substitutes_manual][1]
            people_id = substitute_manual[substitutes_manual][0]
            
            name_dict['check_rule'] = 'manual_substitution'
            name_dict['people_id'] = people_id
            
            if 'STAFF' in role:
                
                name_dict['contacted_staff_counsel'] = True
                name_dict['is_staff_counsel'] = True
                
            
            return name_dict
    
    words = name.split(' ')
    
    # for committee in committee_df.committee_name.values:
        
    #     words_committee = committee.split(' ');
    #     if (committee in name) & (len(words_committee) > 1):
            
    #         target_idx = -1
    #         for word in words:
                
    #             target_idx += 1
    #             if word == words_committee[0]:
                    
    #                 words[target_idx] = committee
    #                 for i in range(1,len(words_committee)-1):
    #                     words.remove(words[target_idx+i])
                
    #                 break
            
            
                
                
    
    count_persons = 0
    
    for word_idx in range(len(words)):
        
        if words[word_idx].endswith("'S"):
            words[word_idx] = words[word_idx][:-2]
            
        elif words[word_idx].endswith("'"):
            words[word_idx] = words[word_idx][:-1]
            
            
    for word in words:
        
        if word == 'SENATOR':
            count_persons += 1
            
    # if count_persons > 1:
        
    #     senators = name.split('SENATOR')
    #     senators = [senator.strip() for senator in senators]
    #     if 'NYS' in senators:
    #         senators.remove('NYS')
    #     if 'STAFF FOR' in senators:
    #         idx = 0
    #         for string in senators:
    #             if string == 'STAFF FOR':
    #                 senators.append(senators[idx] + senators[idx+1])
    #                 break
    #             idx += 1
    #         senators.remove(senators[idx])
    #         senators.remove(senators[idx+1])
        
    #     multiple_rows = []
    #     for senator in senators:
            
    #         #print(senator)
    #         multiple_rows.append(manipulatePartyName(senator, nlp, dictionary, committees))
        
    #     return multiple_rows
    
    for word in words:
        
        subwords = word.split('-')
        
        if (word != '-') & (len(subwords)==2):
             
            if len(subwords[0])*len(subwords[1])>0:
                
                if (subwords[0].lower() in dictionary) | (subwords[1].lower() in dictionary):
                    words.remove(word)
                    words.append(subwords[0])
                    words.append(subwords[1])
                    
                
                        
            #for subword in subwords:
            
    
    count_words = len(words)
    
    # non-dicionary words
    proper_words = []
    for word in words:
        if (word.lower() not in dictionary) & (word.upper() not in drop_words):
            proper_words.append(word)
    # check whether the report string contains exactly the (1-word) last name
    
    
    def matchPersonName(people_df_dict, session_id, words):
        
        contains_member_name = False
        matched_string = ''
        matched_string_idx = 0
        matched_string_position = {}
        for string in words:
            matched_string_idx += 1
            #string = string.replace("-", "")
            if string in set(people_df_dict[session_id].last_name):
                contains_member_name = True
                matched_string = string
                matched_string_position[matched_string_idx] = matched_string
            
        if contains_member_name == True:
            matched_string_idx = max(matched_string_position.keys())
            if (len(matched_string_position) > 1) & (words[min(max(matched_string_position.keys())+1, len(words)-1)] == 'STAFF'):
                matched_string_idx = min(matched_string_position.keys())
            matched_string = matched_string_position[matched_string_idx]
            
        length_multiple_word_lastnames = people_df_dict[session_id].last_name.apply(lambda s: s.split(' ')).apply(len)
        
        if contains_member_name == False:
            
            break_loop = False
            for j in range(2,max(length_multiple_word_lastnames)+1):
                multiple_word_lastnames = people_df_dict[session_id].last_name.apply(lambda s: s.split(' '))[length_multiple_word_lastnames==j]
                for multiple_word_lastname in multiple_word_lastnames:
                    check_last_name = True
                    for i in range(j-1,len(words)):
                        word = words[i]
                        check_last_name = check_last_name & (string in multiple_word_lastname)
                    if check_last_name:
                        matched_string = ' '.join(multiple_word_lastname)
                
                    
        return contains_member_name, matched_string
        
    
    contains_member_name, matched_string = matchPersonName(people_df_dict, session_id, words)
    
    new_session_id = session_id
    
    if (contains_member_name == False) & (len(proper_words)>0):
        
        session_id_idx = np.where(session_ids==session_id)[0][0]
        
        contains_member_name, matched_string = matchPersonName(people_df_dict, session_ids[max(session_id_idx+1, len(session_ids)-1)], words)
        
        if contains_member_name  == True:
            name_dict['check_rule'] = 'contains_exact_name_other_session'
            new_session_id = session_ids[max(session_id_idx+1, len(session_ids)-1)]
            
        elif contains_member_name  == False:
            
            for i in range(1,len(session_ids)-1):
                
                contains_member_name, matched_string = matchPersonName(people_df_dict, session_ids[max(session_id_idx-i, 0)], words)

                if contains_member_name  == True:
                    name_dict['check_rule'] = 'contains_exact_name_other_session'
                    new_session_id = session_ids[max(session_id_idx-i, 0)]
                    break
        
        
    session_id = new_session_id
    contains_committee_name = False
    
    stop_words = set(stopwords.words('english'))
    matched_committee = ''
    for committee in committee_df.committee_name.values:
        
        committee_words = committee.split(' ')
        committee_words = [word for word in committee_words if word.lower() not in stop_words]
        for word in [word for word in words if word.lower() not in stop_words]:
            
            matched_chars = 0
            for committee_word in committee_words:
                if committee_word == word:
                    
                    matched_chars += 1
                    
                if matched_chars > 0:
                        
                    if (len(committee_words) <= 2) & (len(committee_words) == matched_chars):
                        
                        contains_committee_name = True
                        matched_committee = committee
                        break
                    
                    elif  (len(committee_words) > 2) & (len(committee_words)/matched_chars >= 0.75):
                    
                        contains_committee_name = True
                        matched_committee = committee
                        break
    
    if contains_member_name:
        
        if name_dict['check_rule'] != 'contains_exact_name_other_session':
            name_dict['check_rule'] = 'contains_exact_name'
        
        
        inferred_name_text = " ".join(proper_words)
        # doc = nlp(inferred_name)
        
        # # Extract proper names
        # proper_names = []
        # for ent in doc.ents:
        #      if ent.label_ == "PERSON":
        #          proper_names.append(ent.text)
                 
        if len(inferred_name_text)>0:
            inferred_name = returnHumanName(inferred_name_text)
        else:
            name_dict['possible_error'] = True
        
        
        if matched_string in homonim_names_dict[session_id].last_name.values:
        
            name_dict['check_rule'] = 'contains_homonimous_name'
            matched_subdf = homonim_names_dict[session_id].loc[(homonim_names_dict[session_id].last_name == matched_string)]
            roles_subdf = homonim_names_dict[session_id].loc[(homonim_names_dict[session_id].last_name == matched_string) & (homonim_names_dict[session_id].role_id == role_id)]
            
            
            if len(roles_subdf) == 1:
                name_subdf = roles_subdf.copy()
                
            elif (len(inferred_name['first']) > 0) & (sum(roles_subdf.first_name == inferred_name['first'])):
                
                name_subdf = roles_subdf.loc[roles_subdf.first_name == inferred_name['first']]
                
            else:
                
                matched_subdf['similarity'] = matched_subdf.first_name.apply(lambda name: jarowinkler_similarity(inferred_name['first'], name))
                name_subdf = matched_subdf.loc[matched_subdf.similarity == max(matched_subdf.similarity)]
            
        else:
        
            name_subdf = people_df_dict[session_id].loc[people_df_dict[session_id].last_name == matched_string,:]
            
            if matched_string in similar_names_dict[session_id].last_name.values:
                
                name_dict['check_rule'] = 'contains_similar_name'
                
                if name_subdf.first_name.iloc[0] != inferred_name['first']:
                    name_dict['possible_error'] = True
                
            name_subdf = people_df_dict[session_id].loc[people_df_dict[session_id].last_name == matched_string,:]
        
        name_dict['people_id'] = name_subdf.index[0]
        for col in ['name', 'first_name', 'middle_name', 'last_name', 'suffix', 'nickname']:
            
            name_dict[col] = name_subdf.iloc[0][col]
            
        name_dict['full_name'] = returnFullName(name_dict)
        name_dict['full_name_with_nickname'] = returnFullName(name_dict, substitute_nickname = True)
        name_dict['full_name_wo_middle_name'] = returnFullName(name_dict, substitute_nickname = False, drop_middle_name = True)
        name_dict['full_name_with_alternative_nickname'] = returnFullName(name_dict, substitute_nickname = True, drop_middle_name = True, new_nicknames = new_nicknames)
        
        if 'STAFF' in words:
            
            name_dict['contacted_staff_counsel'] = True
    
    
        if len(inferred_name_text) > 0:
            
            
            dist1 = jarowinkler_similarity(returnHumanName(inferred_name_text, False), name_dict['name'].upper())
            dist2 = jarowinkler_similarity(returnHumanName(inferred_name_text, False), name_dict['full_name'].upper())
            dist3 = jarowinkler_similarity(returnHumanName(inferred_name_text, False), name_dict['full_name_with_nickname'].upper())
            dist4 = jarowinkler_similarity(returnHumanName(inferred_name_text, False), name_dict['full_name_wo_middle_name'].upper())
            dist5 = jarowinkler_similarity(returnHumanName(inferred_name_text, False), name_dict['full_name_with_alternative_nickname'].upper())
            if (max([dist1,dist2, dist3, dist4, dist5]) <= 0.9) & (len(proper_words) > 1):
                    
            #if len(inferred_name['middle']) > 0:
                
            #    if inferred_name['middle'] != name_dict['middle_name']:
                    
                name_dict['possible_error'] = True
                
    elif contains_committee_name:
        
        name_dict['check_rule'] = 'contains_committee_name'
        name_dict['committee_name'] = matched_committee
        name_dict['committee_id'] = committee_df.loc[committee_df.committee_name == matched_committee,'committee_id'].values[0]
                
    elif (contains_member_name == False ) & (len(proper_words)>0):
        
        name_dict['check_rule'] = 'contains_proper_name'
        
    else:
        
        
        if (('SENATE' in words) | ('ASSEMBLY' in words)):
        
            
        
            if ('STAFF' in words) | ('COUNSEL' in words):
               
                name_dict['contacted_staff_counsel'] = True
                name_dict['is_staff_counsel'] = True
               
                # Process the phrase with spaCy
                # doc = nlp(name)
               
                # # Extract proper names
                # proper_names = []
                # for ent in doc.ents:
                #     if ent.label_ == "PERSON":
                #         proper_names.append(ent.text)
               
                name_dict['check_rule'] = 'contains_staff_general'
                
                if len(proper_words) > 0:
                  
                    name_dict['is_staff_counsel'] = True
                    name_dict = name_dict | returnHumanName(proper_words[0])
                    
                    name_dict['check_rule'] = 'contains_staff_name'
                    
                
                
                else:
                    if 'MAJORITY' in words:
                       
                        name_dict['majority_body'] = True
                        
                        name_dict['check_rule'] = 'contains_majority_general'
                        
                    elif ('MINORITY' in words) | ('REPUBLICAN' in words):
                       
                        name_dict['minority_body'] = True
                        
                        name_dict['check_rule'] = 'contains_minority_general'
                        
                    else:
                       
                        name_dict['entire_body'] = True
                        
                        name_dict['check_rule'] = 'contains_body_general'
               
            if 'MAJORITY' in words:
               
                name_dict['majority_body'] = True
                
                name_dict['check_rule'] = 'contains_majority_general'
                
            elif ('MINORITY' in words) | ('REPUBLICAN' in words):
               
                name_dict['minority_body'] = True
                
                name_dict['check_rule'] = 'contains_minority_general'
                
            else:
               
                name_dict['entire_body'] = True
                
                name_dict['check_rule'] = 'contains_body_general'
            
        elif ('ALL' in words) | ('ENTIRE' in words):
        
            name_dict['entire_body'] = True
        
    # #elif 'MAJORITY' in words:
        
        
        
    # elif count_words == 1:
        
    #     if (name == 'STAFF') | (name == 'COUNSEL'):
            
    #         name_dict['contacted_staff_counsel'] = True
    #         name_dict['is_staff_counsel'] = True
            
    #     elif (name == 'ALL') | (name == 'MEMBERS'):
            
    #         name_dict['entire_body'] = True
            
    #     elif name in committees:
            
    #         name_dict['committee'] = name
    #         name_dict['entire_committee'] = True
            
    #     else:
            
    #         name_dict = name_dict | returnHumanName(name)
    
    # elif 'DELEGATION' in words:
        
    #     name_dict['other'] = True
        
    # else:
        
    #     if ('STAFF' in words):
            
    #         name_dict['contacted_staff_counsel'] = True
            
    #         if name in committees:
                
    #             name_dict['committee'] = name
                
    #         elif 'MAJORITY' in words:
                
    #             name_dict['majority_body'] = True
                
    #         else:
    #             words.remove('STAFF')
    #             # Process the phrase with spaCy
                
    #             if len(words) == 1:
                    
    #                 return name_dict | returnHumanName(words[0])
                
    #             new_name = ''
                
    #             for word in words:
                    
    #                 new_name = new_name + word.capitalize() + ' '
                    
    #             new_name = new_name.rstrip()
                
    #             doc = nlp(new_name)
                
    #             # Extract proper names
    #             proper_names = []
    #             for ent in doc.ents:
    #                 if ent.label_ == "PERSON":
    #                     proper_names.append(ent.text)
                
    #             if len(proper_names) > 0:
                    
    #                 name_dict['is_staff_counsel'] = True
    #                 name_dict = name_dict | returnHumanName(proper_names[0])
            
    #             return name_dict
    #     else:
            
    #             new_name = ''
                
    #             for word in words:
                    
    #                 new_name = new_name + word.capitalize() + ' '
                    
    #             new_name = new_name.rstrip()
                
    #             doc = nlp(new_name)
                
    #             # Extract proper names
    #             proper_names = []
    #             for ent in doc.ents:
    #                 if ent.label_ == "PERSON":
    #                     proper_names.append(ent.text)
                
    #             if len(proper_names) > 0:
                    
    #                 name_dict['is_staff_counsel'] = True
    #                 name_dict = name_dict | returnHumanName(proper_names[0])
            
    #             return name_dict
            
    
    return name_dict
    
    
        
def returnHumanName(full_name, as_dict = True):
    
    human_name = HumanName(full_name)
    
    if (len(human_name.first) > 0) & (len(human_name.last) == 0):
        
        human_name.last = human_name.first
        human_name.first = ''
        
    if as_dict == False:
        
        return human_name.full_name
    
    else:   
        return human_name.as_dict()
    
    return human_name

