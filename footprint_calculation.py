# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import time

import pathlib
import os
import json
import zipfile

import emoji
from fuzzywuzzy import process, fuzz
import re

from fractions import Fraction

################################## Load the data
df_products = pd.read_csv('/Users/CristaVillatoro/Desktop/tahini-tensor-student-code 2/FINAL-project/streamlit/data/footprint_products_agribalyse.csv', index_col=0)

def load_dataset(silent=False):
    dataset_file_names = [
        'recipes_raw_nosource_ar.json',
        'recipes_raw_nosource_epi.json',
        'recipes_raw_nosource_fn.json',
    ]
    
    dataset = []

    for dataset_file_name in dataset_file_names:
        dataset_file_path = f'recipes_raw/{dataset_file_name}'

        with open(dataset_file_path) as dataset_file:
            json_data_dict = json.load(dataset_file)
            json_data_list = list(json_data_dict.values())
            dict_keys = [key for key in json_data_list[0]]
            dict_keys.sort()
            dataset += json_data_list

            if silent == False:
                print(dataset_file_path)
                print('===========================================')
                print('Number of examples: ', len(json_data_list), '\n')
                print('Example object keys:\n', dict_keys, '\n')
                print('Example object:\n', json_data_list[0], '\n')
                print('Required keys:\n')
                print('  title: ', json_data_list[0]['title'], '\n')
                print('  ingredients: ', json_data_list[0]['ingredients'], '\n')
                print('  instructions: ', json_data_list[0]['instructions'])
                print('\n\n')

    return dataset  
dataset_raw = load_dataset() 

################################ Processing the data
# Filters out recipes which don't have either title or ingredients or instructions.
def recipe_validate_required_fields(recipe):
    required_keys = ['title', 'ingredients', 'instructions']
    
    if not recipe:
        return False
    
    for required_key in required_keys:
        if not recipe[required_key]:
            return False
        
        if type(recipe[required_key]) == list and len(recipe[required_key]) == 0:
            return False
    
    return True

dataset_validated = [recipe for recipe in dataset_raw if recipe_validate_required_fields(recipe)]

############################### Converting recipes into strings
# Adding Landmarks
STOP_WORD_TITLE = emoji.emojize(':books:')
STOP_WORD_INGREDIENTS = emoji.emojize('\n:hot_pepper:\n\n')
STOP_WORD_INSTRUCTIONS = emoji.emojize('\n:cook:\n\n')

# Converts recipe object to string (sequence of characters) for later usage in RNN input.
def recipe_to_string(recipe):
    # This string is presented as a part of recipes so we need to clean it up.
    noize_string = 'ADVERTISEMENT'
    
    title = recipe['title']
    ingredients = recipe['ingredients']
    instructions = recipe['instructions'].split('\n')
    
    ingredients_string = ''
    for ingredient in ingredients:
        ingredient = ingredient.replace(noize_string, '')
        if ingredient:
            ingredients_string += f'• {ingredient}\n'
    
    instructions_string = ''
    for instruction in instructions:
        instruction = instruction.replace(noize_string, '')
        if instruction:
            instructions_string += f'▪︎ {instruction}\n'
    
    return f'{STOP_WORD_TITLE}{title}\n{STOP_WORD_INGREDIENTS}{ingredients_string}{STOP_WORD_INSTRUCTIONS}{instructions_string}'

dataset_stringified = [recipe_to_string(recipe) for recipe in dataset_validated]

#################################### Removing large recipes
DEBUG = False
DEBUG_EXAMPLES = 10

# A limit of 2000 characters for the recipes will cover 80+% cases.
# Decided to train RNN with this maximum recipe length limit.
MAX_RECIPE_LENGTH = 2000

if DEBUG:
    MAX_RECIPE_LENGTH = 500

def filter_recipes_by_length(recipe_test):
    return len(recipe_test) <= MAX_RECIPE_LENGTH 

dataset_filtered = [recipe_text for recipe_text in dataset_stringified if filter_recipes_by_length(recipe_text)]

TOTAL_RECIPES_NUM = len(dataset_filtered)

################################# Dataset parameters
# create empty lists to store the data
titles = []
ingredients = []
procedures = []

# loop through the list of recipes
for recipe in dataset_filtered:
    # split the recipe into title, ingredients, and procedure
    recipe = recipe.split("\n\n")
    title = recipe[0]
    ingreds = recipe[2]
    proc = recipe[4]
    
    # append the data to the corresponding list
    titles.append(title)
    ingredients.append(ingreds)
    procedures.append(proc)
# create a data frame using the lists
df_recipes = pd.DataFrame({'Title': titles, 'Ingredients': ingredients, 'Procedure': procedures})

################################ Caluclation of the CO2 emissions
# Extracting the ingredients list of the recipe
df_extracted = df_products.set_index('Food Subgroup')
df_extracted = df_extracted.loc[['Special Products', 'Herbs', 'Miscellaneous Ingredients', 'Spices','Sels', 'Condiments','Fruits', 'Nuts And Oilseeds',
       'Vegetables', 'Potatoes And Other Tubers', 'Legumes', 'Cheeses','Creams And Cream Specialties',
       'Milks', 'Butters', 'Vegetable Oils And Fats', 'Other Fats',
       'Fish Oils', 'Margarines', 'Pasta, Rice And Cereals',
       'Flours And Pastry',
       'Chocolates And Chocolate Products', 
       'Jams And Similar', 'Sugars, Honeys And Similar',
       'Fish And Seafood Products', 'Cooked Meats', 'Raw Meats',
       'Raw Fish', 'Deli Meats', 'Cooked Fish', 'Other Meat Products',
       'Meat Substitutes', 'Cooked Shellfish', 'Raw Shellfish', 'Eggs',
       'Deli Substitutes']]
df_extracted.reset_index(inplace=True)

######################################### Picking one recipe to calculate the environmental footprint

recipe_string = dataset_filtered[70000]

# Split the recipe string into lines
lines = recipe_string.split('\n')


# Create an empty list to store the ingredients
ingredients = []

# Loop through each line and check if it contains an ingredient
for line in lines:
    # Get the title of the recipe from the first line
    title = lines[0]    
    if '• ' in line:
        ingredient = line.strip().replace('• ', '')
        ingredients.append(ingredient)

# get the list of ingredients from the recipe
ingredients_list = ingredients

# Define a list of known unit of measurement keywords
unit_keywords = ['clove', 'cloves', 'small', 'large', 'gram', 'g', 'kilogram', 'kg', 'milligram', 'mg', 'ounce', 'oz', 'pound', 'lb', 'tablespoon', 'tbsp', 'teaspoon', 'tsp', 'cup', 'can', 'jar', 'bottle', 'package', 'pinch', 'piece', 'slice']

# create a dictionary to store the matched LCI names, quantity and unit of measurement
lcis = {}
title = title
# iterate over the ingredients list and match each ingredient to the LCI names in the dataset
for ingredient in ingredients_list:
    ingredient = str(ingredient)
    print(f"ingredient: {ingredient}")
    df_extracted['LCI Name'] = df_extracted['LCI Name'].astype(str)
    # get the best match using the fuzzywuzzy library
    best_match = process.extractOne(ingredient, df_extracted['LCI Name'])
    print(f"best match: {best_match}")
    # get the index of the best match
    index = df_extracted[df_extracted['LCI Name'] == best_match[0]].index[0]
    # extract the quantity and unit of measurement from the ingredient string
    quantity, unit = None, None
    for word in ingredient.split():
        if any(keyword in word.lower() for keyword in unit_keywords):
            unit = word.lower()
            quantity = ingredient.split(unit)[0].strip()
            break
    # store the matched LCI name, its corresponding Climate change value, quantity and unit of measurement in the dictionary
    lcis[df_extracted.at[index, 'LCI Name']] = {'Climate change (kg CO2 eq/kg product)': df_extracted.at[index, 'Climate change (kg CO2 eq/kg product)'],
                                      'Single EF 3.1 score (mPt/kg product)': df_extracted.at[index, 'Single EF 3.1 score (mPt/kg product)'],
                                      'Packaging Material': df_extracted.at[index, 'Packaging Material'],
                                      'Quantity': quantity,
                                      'Unit': unit}

lcis_df = pd.DataFrame.from_dict(lcis, orient='index')
lcis_df['Title'] = [title] * len(lcis_df)
lcis_df.loc[len(lcis_df)] = np.nan
lcis_df['original_ingredients'] = ingredients

###################################### FINAL calculation

conversion_factors = {
    'clove': 0.005,
    'cloves': 0.005,
    'small': 0.01,
    'large': 0.05,
    'gram': 0.001,
    'g': 0.001,
    'kilogram': 1,
    'kg': 1,
    'milligram': 0.000001,
    'mg': 0.000001,
    'ounce': 0.028,
    'oz': 0.028,
    'pound': 0.453592,
    'lb': 0.453592,
    'tablespoon': 0.015,
    'tbsp': 0.015,
    'teaspoon': 0.005,
    'tsp': 0.005,
    'cup': 0.24,
    'can': 0.4,
    'jar': 0.5,
    'bottle': 0.5,
    'package': 1,
    'pinch': 0.001,
    'piece': 1,
    'slice': 0.03,
}

def convert_quantity(quantity, unit):
    if unit == 'egg':
        unit = 'small'
    if isinstance(quantity, str):
        # check if quantity is numeric with possible fractions
        if all(char.isdigit() or char == '/' or char == '.' for char in quantity):
            parts = quantity.split(' ')
            total = 0
            for part in parts:
                if '/' in part:
                    num, denom = part.split('/')
                    total += float(num) / float(denom)
                else:
                    total += float(part)
            return total * conversion_factors.get(unit, 0)
        else:
            return 0
    elif isinstance(quantity, (int, float)):
        return quantity * conversion_factors.get(unit, 0)
    else:
        return 0
    
lcis_df['quantity_kg'] = lcis_df.apply(lambda row: convert_quantity(row['Quantity'], row['Unit']), axis=1)

# Calculation of CO2 emissions for each ingredient
lcis_df['impact'] = lcis_df['quantity_kg'] * lcis_df['Climate change (kg CO2 eq/kg product)']

# Total CO2 emissions per recipe
total_impact = lcis_df['impact'].sum()
print(f'The total impact of a recipes is: {total_impact}')

# Total single EF per recipe
total_single_ef = sum(row['Single EF 3.1 score (mPt/kg product)'] * row['quantity_kg']
                      for index, row in lcis_df.dropna().iterrows())
print(f'The total impact of a recipes is: {total_single_ef}')