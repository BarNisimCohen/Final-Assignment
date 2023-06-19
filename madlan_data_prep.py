# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 17:24:18 2023

@author: berber
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime,timedelta

file_name = "output_all_students_Train_v10.xlsx"
df = pd.read_excel(file_name)

def general_clean(df):
    # Check and remove leading/trailing spaces in column names
    df.columns = df.columns.str.strip()
    df['City']=df['City'].str.replace(' נהרייה','נהריה')
    df['City']=df['City'].str.replace('נהרייה','נהריה')
    df['City']= df['City'].str.replace(' שוהם','שוהם')

    #column 'num_of_images' as numeric
    df['num_of_images'].fillna(0, inplace=True)
    df['num_of_images'].astype(int)
    #column 'number_in_street' as numeric
    df['number_in_street'].fillna(0, inplace=True)
    df['number_in_street'].replace([np.inf, -np.inf], 0, inplace=True)
    df['number_in_street'] = pd.to_numeric(df['number_in_street'], errors='coerce')
    return df

def extract_numeric_value(df, column):
    df[column] = df[column].astype(str).apply(lambda x: float(re.sub(r'[^0-9]', '', x)) if re.search(r'[0-9]', x) else None)
    return df
def extract_numeric_room(df, column):
    df[column] = df[column].astype(str).apply(lambda x: float(re.sub(r'[^0-9.]', '', x.replace(',', '.'))) if re.search(r'[0-9.]', x) else None)
    df = df[df[column] != 0]
    return df
    
def remove_rows_with_missing_values(df, column_name):
    df = df.dropna(subset=['price'] , axis=0, inplace=True)
    return df

def drop_punctuation(df, column):
    df[column] = df[column].astype(str).str.replace(r'[^\w\s]', '', regex=True)
    return df

def floor_split(df, column):
    df[column] = df[column].fillna('')  # Fill missing values with an empty string
    
    # Extract floor number from "קומה 6" format
    df['floor'] = df[column].str.extract(r'קומה\s+(\d+)')
    df['floor'] = df['floor'].astype(float)
    
    # Extract floor and total floor numbers from "קומה 6 מתוך 8" format
    floor_total_floor = df[column].str.extract(r'קומה\s+(\d+)\s+מתוך\s+(\d+)')
    floor_total_floor.columns = ['floor', 'total_floors']
    floor_total_floor = floor_total_floor.astype(float)
    
    # Update the 'floor' column where the total floor number is available
    df['floor'].fillna(floor_total_floor['floor'], inplace=True)
    
    # Update the 'TOTAL_FLOORS' column where the total floor number is available
    df['total_floors'] = floor_total_floor['total_floors']
    
    # Set floor and total floor numbers for 'קרקע' and 'מרתף' cases
    df.loc[df[column].str.contains('קרקע'), ['floor', 'total_floors']] = 0, 0
    df.loc[df[column].str.contains('מרתף'), ['floor', 'total_floors']] = -1, 0
    df['floor'].fillna(0.0, inplace=True)
    df['total_floors'].fillna(0.0, inplace=True)
    

def entrance_date(df, column):
    today = datetime.now().date()
    def calculate_category(date_str):
        try:
            entrance_date = pd.to_datetime(date_str, format='%d/%m/%Y').date()
            day_diff = (entrance_date - today).days
        
            if day_diff < 6 * 30:  # Less than 6 months (approximated as 30 days per month)
                return 'less_than_6 months'
            elif 6 * 30 <= day_diff < 12 * 30:  # Between 6 months and 12 months
                return 'months_6_12'
            else:  # More than 12 months
                return 'above_year'
        except:
            pass

    mapping = {
        np.nan: 'not_defined',
        "לא צויין": "not_defined",
        "לא צוין": "not_defined",
        "גמיש": 'flexible',
        "מיידי": 'less_than_6 months',
        "מידי": 'less_than_6 months'
    }

    df['entrance_date'] = df[column].map(mapping)
    df['entrance_date'] = np.where(df['entrance_date'].isna(), df[column].apply(calculate_category), df['entrance_date'])
    return df
    
def replace_values_with_zero(df, column, search_terms_false):
    df[column] = df[column].apply(lambda x: 0 if any(term.lower() in str(x).lower() for term in search_terms_false) else x)
    return df
def replace_values_with_one(df, column, search_terms_true):
    df[column] = df[column].apply(lambda x: 1 if any(term.lower() in str(x).lower() for term in search_terms_true) else x)
    return df

def encode_category(df,column_name):
    unique_cities = df[column_name].unique()
    lst_encoding = {city: i + 1 for i, city in enumerate(unique_cities)}
    df[column_name+'-enc'] = df[column_name].map(lst_encoding)
    return df

    
def prepare_data(df) :
    try:
        general_clean(df)
    except:
        pass
    try: #remove assets without price
        remove_rows_with_missing_values(df, 'price')
    except:
        pass
    try: # price is numeric type
        extract_numeric_value(df, 'price')
    except:
        pass
    try: #area in numeric
        extract_numeric_value(df, 'Area')
    except:
        pass
    try: #rooms in numeric
        extract_numeric_room(df, 'room_number')
        df['room_number'].fillna(0.0, inplace=True)
    except:
        pass
    try: #clean the text from punctuation
        drop_punctuation(df, 'Street')
        drop_punctuation(df, 'city_area')
        drop_punctuation(df,'description')
    except:
        pass
    try: #add column 'floof'&'total_floor'
        floor_split(df, 'floor_out_of')
    except:
        pass
    try: #ad column 'entrance_date'
        entrance_date(df, 'entranceDate')
    except:
        pass
    try: #Boolean
        search_terms_false = ['False', 'אין', 'no','לא צויין','אין חניה','0','לא']
        search_terms_true = ['True', 'יש', 'yes', 'חלקי','נגיש לנכים ','1','כן']
        replace_values_with_zero(df, 'hasMamad', search_terms_false)
        replace_values_with_one(df, 'hasMamad', search_terms_true)
        replace_values_with_zero(df, 'hasBalcony', search_terms_false)
        replace_values_with_one(df, 'hasBalcony', search_terms_true)
        replace_values_with_zero(df, 'handicapFriendly', search_terms_false)
        replace_values_with_one(df, 'handicapFriendly', search_terms_true)
        replace_values_with_zero(df, 'furniture', search_terms_false)
        replace_values_with_one(df, 'furniture', search_terms_true)
        replace_values_with_zero(df, 'hasElevator', search_terms_false)
        replace_values_with_one(df, 'hasElevator', search_terms_true)
        replace_values_with_zero(df, 'hasParking', search_terms_false)
        replace_values_with_one(df, 'hasParking', search_terms_true)
        replace_values_with_zero(df, 'hasBars', search_terms_false)
        replace_values_with_one(df, 'hasBars', search_terms_true)
        replace_values_with_zero(df, 'hasStorage', search_terms_false)
        replace_values_with_one(df, 'hasStorage', search_terms_true)
        replace_values_with_zero(df, 'hasAirCondition', search_terms_false)
        replace_values_with_one(df, 'hasAirCondition', search_terms_true)
    except:
        pass
    try:
        df.fillna(0, inplace=True)
    except:
        pass
    try:
        encode_category(df,'City')
        encode_category(df,'type')
        encode_category(df,'condition')
        encode_category(df,'entrance_date')
        
    except:
        pass

    return df

prepare_data(df)
