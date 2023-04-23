#import tensorflow as tf
import pandas as pd
import numpy as np
from fractions import Fraction
import altair as alt

import re
from fuzzywuzzy import process, fuzz
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import plotly.offline as py

import streamlit as st
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode, JsCode
from streamlit_option_menu import option_menu
from streamlit_card import card


import base64
from PIL import Image
import time
import webbrowser 


################################# Load the data

# Load the emissions dataset for the products
df_products = pd.read_csv('data/footprint_products_agribalyse.csv', index_col=0)
# Load the full recipes
df_full_recipe = pd.read_csv('data/recipes_full.csv')

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

############################### Sidebar

with st.sidebar:
    selected2 = option_menu(menu_title='Main menu', options=["Home","Personal cookbook", "Climate Calculator App", "Environmental footprint of products dataset", "New recipes generation using Recurrent Neural Networks"], 
    icons=['house', "book", 'flower1','bar-chart-line', 'gear'], 
    menu_icon="cast", default_index=0, styles={
        "container": {"padding": "0!important", "background-color": "##8fbc8f"},
        "icon": {"color": "black", "font-size": "15px"}, 
        "nav-link": {"font-size": "15px", "text-align": "left", "margin":"0px", "--hover-color": "##8fbc8f"},
        "nav-link-selected": {"background-color": "#8fbc8f"},
    })

################################# Home 
if selected2 == 'Home':
    # Create three columns for the three images
    st.title('CookWise')
    st.header("The climate calculator for your kitchen")
    col1, col2 = st.columns(2)
    with col1:
        image1 = st.image("data/Food.png", use_column_width=True)
    with col2:
        st.write("Salads, soups, meat or cakes? Using `CookWise app` you can calculate how climate-friendly your own recipes are. :seedling::green_heart:")
        st.markdown('''  
        \n In this tool you can find a dataset of more than 100 thousand different cooking recipes, 
        the calculation of the environmental footprint of each of them, the environmental footprint of a dataset of products available in a supermarket,
        and a LSTM (Long short-term memory) Recurrent Neural Network model that generates cooking recipes.
        ''')
        st.markdown('''  
        \n##### Created by Crista Villatoro 
        ''')

        st.markdown("---")

    st.markdown('''
    Start cooking smarter with CookWise today! This interactive calculator can help you make informed decisions. 
    In this section, simply select a food product to see how much CO₂ emissions produces. You can also compare it with 
    other food products of the same category or change the parameter you would like to display.\n\n 
    ''')

    with st.expander('More information about the parameters that can be selected'):
        st.markdown("""
        1. Climate change: The best known indicator, corresponds to the modification of the climate, affecting the global ecosystem. Given 
        in CO2 eqivalent per kilogram of product. 
        \n2. CO2 emissions units: A carbon dioxide equivalent or CO2 equivalent, abbreviated as CO2-eq is a metric measure 
        used to compare the emissions from various greenhouse gases on the basis of their global-warming potential (GWP), 
        by converting amounts of other gases to the equivalent amount of carbon dioxide with the same global warming potential.
        Source [link](https://ec.europa.eu/eurostat/statistics-explained/index.php?title=Glossary:Carbon_dioxide_equivalent)
        \n3. Single EF Score: It is recommended by the European Commission, calculated with weighting factors for 
        each of the indicators; the weighting takes into account both the relative robustness of each of these indicators and the environmental challenges. 
        \n4. Water use: Corresponds to the consumption of water and its depletion in certain regions. This category takes into account scarcity 
        (it has more impact of consuming a liter of water in Morocco than in Norway).
        \n5. Land use: Land is a finite resource, which is shared between "natural" (forest), productive (agriculture) and urban environments.
          Land use and habitats largely determine biodiversity. This category therefore reflects the impact of an activity on land degradation, 
          with reference to "the natural state".
        """)

        # Sidebar 
    st.sidebar.markdown(
        "**Note:** The following selection interacts with the Fig.1."
    )
    st.sidebar.markdown("## Selection")

    # Create a selectbox in the sidebar for the user to choose a product
    product_options = df_products["LCI Name"].tolist()
    selected_product = st.sidebar.selectbox("Choose a product:", product_options, index=product_options.index("Garlic, fresh"))

    # Filter the dataframe to only show the selected product
    selected_row = df_products.loc[df_products["LCI Name"] == selected_product]
    food_subgroup = selected_row["Food Subgroup"].values[0]
    subgroup_df = df_products.loc[df_products["Food Subgroup"] == food_subgroup]

    # Create radio buttons in the sidebar for the user to choose a metric
    metric_options = ["CO2 Emissions", "Single EF Score", 'Water resource depletion',"Land Use"]
    selected_metric = st.sidebar.radio("Choose a parameter:", metric_options)
    metric_dict = {
        "CO2 Emissions": "Climate change (kg CO2 eq/kg product)",
        "Single EF Score": "Single EF 3.1 score (mPt/kg product)",
        "Water resource depletion": "Water resource depletion (m3 depriv./kg product)",
        "Toxicological effects on human health: non-carcinogens": "Toxicological effects on human health: non-carcinogens (CTUh/kg product)",
        "Toxicological effects on human health: carcinogenic substances": "Toxicological effects on human health: carcinogenic substances (CTUh/kg product)",
        "Land Use": "Land use (Pt/kg product)"
    }
    selected_metric = metric_dict[selected_metric]
    
    st.write("\n\n")
    st.write(f"##### :white_check_mark: Your selected product is: {selected_product}")
    value = selected_row[selected_metric].values[0]

    st.write(f"{selected_product} has a {selected_metric} of {round(value, 3)}")

    st.write(f"""In the following graph you can observe the {selected_metric} for {food_subgroup}, which is the food category
    of your selected product.
            """)


    st.write(f"###### Fig.1: {selected_metric} for {food_subgroup}")
    # Create a bar chart of the selected metric for the selected product and all other products in the same food subgroup
    fig = px.bar(subgroup_df, x="LCI Name", y=selected_metric, color_discrete_sequence=["#8fbc8f"])
    fig.update_xaxes(tickangle=90)
    st.plotly_chart(fig)

    with st.expander('Information about the dataset:'):
        st.markdown('''
        The above data belongs to the french AGRIBALYSE® program. Since 2013, AGRIBALYSE® provides references data on 
        the environmental impacts of agricultural and food products through a database built according to the Life Cycle Analysis (LCA) methodology. 
        \nAccording to AGRIBALYSE® documentation, this data is built at first for french situation and food market. But it can be suitable as « first approach proxy » for European countries.
        ''')
        st.write('Source of the dataset: [link](https://doc.agribalyse.fr/documentation-en/agribalyse-data/data-access)')



############################## Climate app

if selected2 == 'Climate Calculator App':
    # Set page configuration
    st.markdown("## Climate calculator - Environmental footprint of a cookig recipe")

    st.markdown("---")
    st. markdown("""
        In this section you can calculate the environmental footprint of your own recipe by adding the ingredients, quantities and units.\n
        After doing that you will find a table with the CO2 emissions (kg CO2 eq/kg product) of each ingredient and the total.
        """)
    product_options = df_products["LCI Name"].tolist()
    col1, col2, col3 = st.columns(3)
    with col1:
        product1 = st.selectbox('Select product 1', df_products['LCI Name'].unique(), index=product_options.index("Chicken, meat, raw"))
        product2 = st.selectbox('Select product 2', df_products['LCI Name'].unique(), index=product_options.index("Rice, brown, raw"))
        product3 = st.selectbox('Select product 3', df_products['LCI Name'].unique(), index=product_options.index("Tomato, off season, raw"))
    with col2:
        quantity1 = st.number_input('Quantity of product 1', value=200)
        quantity2 = st.number_input('Quantity of product 2', value=250)
        quantity3 = st.number_input('Quantity of product 3', value=400)
    with col3:
        unit1 = st.selectbox('Unit of product 1', ['g', 'ml', 'oz', 'pound'])
        unit2 = st.selectbox('Unit of product 2', ['g', 'ml', 'oz', 'pound'])
        unit3 = st.selectbox('Unit of product 3', ['g', 'ml', 'oz', 'pound'])

    # Add button to allow users to add more products
    if st.button('Add another product'):
            with col1:
                product = st.selectbox('Select product', df_products['LCI Name'].unique())
            with col2:
                quantity = st.number_input('Quantity of product')
            with col3:
                unit = st.selectbox('Unit of product', ['g', 'ml', 'oz', 'pound'])


    # Display table of selected products with quantities, units, and emissions
    selected_products = pd.DataFrame({
        'Product': [product1, product2, product3],
        'Quantity': [quantity1, quantity2, quantity3],
        'Unit': [unit1, unit2, unit3],
        'Emissions': [0, 0, 0]
    })
    if 'product' in locals():
        selected_products = selected_products.append({
            'Product': product,
            'Quantity': quantity,
            'Unit': unit,
            'Emissions': 0
        }, ignore_index=True)
    for i, row in selected_products.iterrows():
        product = row['Product']
        quantity = row['Quantity']
        unit = row['Unit']

        # Convert quantity to kg
        if unit == 'g':
            quantity /= 1000
        elif unit == 'ml':
            # Assume water density of 1 g/ml
            quantity /= 1000
        elif unit == 'oz':
            quantity *= 0.0283495
        elif unit == 'pound':
            quantity *= 0.453592

        # Look up CO2 emissions for product
        emissions = df_products[df_products['LCI Name'] == product]['Climate change (kg CO2 eq/kg product)'].iloc[0] * quantity
        selected_products.at[i, 'Emissions'] = emissions

    cola_1, cola_2 = st.columns(2)
    with cola_1:
        # Display table of selected products with quantities, units, and emissions
        st.write('### Selected Products')
        st.markdown("""
        In the following table you can observe the CO2 emissions(kg CO2 eq/kg product) of your selected products,
        according to the unit and quatity you have imputed. 
        """)
        st.write(selected_products)
    with cola_2:
        # Display the GIF with a specific width
        st.image("data/salad.png")

    # Calculate total CO2 emissions
    total_emissions = selected_products['Emissions'].sum()

    # Display total CO2 emissions
    st.metric('Total CO2 Emissions', f'{total_emissions:.2f} (kg CO2 eq/kg product)')

    # Display CO2 progress bar
    st.write("CO2-Bilanz")
    st.write("402g CO2 pro Standard-Portion.")
    progress = st.progress(total_emissions / 4.02)  # Scale progress bar to match 402g CO2 per standard portion


############################################ RNN recipes generation

if selected2 == 'New recipes generation using Recurrent Neural Networks':
    
    st.markdown("## Recipe Generation Using Recurrent Neural Network Models")

    st.markdown("---")
    st.header("Recipe Generation")
    input_letters = st.multiselect('Select up to 2 keywords for generating the recipe:',['Soup', 'Salad', 'Meat', 'Cake','Vegetable', 'Mushroom', 'Apple', 'Berries', 'Chicken', 'Tofu', 'Chickpeas'])
    st.write('You selected:', input_letters)
    
    from recipe_generator import model, generate_combinations
    progress_text =''
    if st.button('Generate Recipes'):
        progress_text = st.write('### Operation in progress...\n #### Please wait! :hourglass_flowing_sand: ')
        # progress_text = "Operation in progress. Please wait."
        my_bar = st.progress(0, text=progress_text)

        for percent_complete in range(100):
            time.sleep(1.7)
            my_bar.progress(percent_complete + 1, text=progress_text)
        
        st.success("Recipe generation is almost complete!")
        st.write(generate_combinations(model, input_letters))

    ####### Model explanation 
    tab1, tab2, tab3 = st.tabs(["Video", "RNN generated recipes","Model Explanation"])
    with tab1:
        video_file = open('final.mov', 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)

    with tab2:
        st.write("#### Generated Recipes using the LSTM-RNN model")
        st.markdown('''
        The following recipes are some examples of the text generation picking up to 2 keywords
        in the above set.
        ''')
        st.write('##### Recipe 1:')
        st.markdown('''
        📚Mushroom Plantains: Pretzel Doughnuts

🌶️

• 1 small bunch basil • 1 stick salted butter, softened • 1/2 cup light brown sugar 
• 1 1/3 cups sugar • 4 tablespoons unsalted butter • 3 tablespoons brown sugar • 1/4 teaspoon salt 
• 2 eggs • 2 tablespoons brandy • 1 1/2 tablespoons unsalted butter, at room temperature 
• 3 cups all-purpose flour • 2 tablespoons sugar • 1/2 teaspoon baking powder • 1/4 teaspoon salt 
• 6 tablespoons (1 stick) cold butter, cut into cubes

🧑‍🍳

▪︎ Special equipment: a 10-inch tube pan or a pan. Spray an 8 by 8-inch baking dish with nonstick cooking spray. 
Place the pumpkin puree eggs, maple syrup, milk, lemon juice, ginger, cinnamon and salt into a large mixing bowl. 
Add a pinch of salt and almond extracts, salt, and cinnamon in the deep bowl and mix well. Add the honey and a pinch of salt and mix well. 
Add the egg mixture to the dry ingredients and mix until the mixture resembles coarse crumbs. 
Sprinkle in flour mixture and stir a few second tilted on the mixer until t
        
        ''')
        st.write('##### Recipe 2:')
        st.markdown('''
        📚Tofu Powder

🌶️

• 2 tablespoons unsalted butter, melted, plus more for greasing • 1 large egg • 1 stick (1/2 cup) sugar 
• 1 teaspoon kosher salt • 1 1/2 cups heavy cream • 1 cup frozen cherries or other rolled oats 
• 2 cups peeled, seeded and coarsely chopped fresh chives • 1 cup minced celery stalk • 1/2 cup pecan halves 
• 1/4 cup chopped prallions • 1/4 cup honey

🧑‍🍳

▪︎ Watch how to make this recipe. ▪︎ Preheat the oven to 400 degrees F. ▪︎ Open the toasted bread slices and put into a roasting pan over the hot oil. 
Cook the chocolate and butter over medium heat until the chocolate is melted, about 4 minutes. 
▪︎ For the remaining chips and butter in a few minutes the chocolate may be eaten add salt, to taste. 
▪︎ To make the glaze, check the peanut butter, and place the bread slices in a bowl. ▪︎ Place the marshmallow chunks in a food processor. Add the ground pork shoulder, 
cocoa powder and water and blend until smooth. Add the chopped chipotle mayonnaise, sour cream, cilantro, and sou
        
        ''')
    
    with tab3:
        st.write("#### Model Explanation")
        st.markdown('##### Long short-term memory (LSTM) - Recurrent Neural Network (RNN)')
        st.image('data/LSTM_diagram.png')
        with st.expander('Build the model:'):
            st.markdown('''
    It was used a Keras Sequential to define the model:
    \n1. tf.keras.layers.Embedding - the input layer (a trainable lookup table that will map the numbers of each character to a vector with embedding_dim dimensions)
    \n2. tf.keras.layers.LSTM - a type of RNN with size of 1024 units
    \n3. tf.keras.layers.Dense - the output layer with the size of the tokenized characters
    \n4. 12 epochs were used to get the final output

            ''')
            st.image('data/loss.png')
        with st.expander('Animation of LSTM-RNN'):
            st.image('https://static.wixstatic.com/media/3eee0b_969c1d3e8d7943f0bd693d6151199f69~mv2.gif')
            st.write('Source of the image: [link](https://www.simplilearn.com/tutorials/deep-learning-tutorial/rnn)')


    


################################### Personal cookbook
if selected2 == "Personal cookbook":
    st.markdown("## Personal cookbook")
    st.markdown("---")

    colp_1, colp_2 = st.columns(2)

    with colp_1:
        st. markdown("""
        In this section you can find a personal cookbook with more than 100 thousand recipes! You can select up to
        5 ingredients to look for a match of a recipe in the dataset.\n
        Once you selected the ingredients, don't forget to click in the recipe and you will have the option to download it
        and to get information about the CO2 emissions of each ingredient and the environmental footprint of the complete recipe. 
        """)
        with st. expander('Information about the recipe dataset and the data preprocessing.'):
            st. markdown("""
            This app makes use of Ryan Lee's Recipe Box dataset which contains 125,000 scraped recipes from 3 websites (Foodnetwork.com, 
            Epicurious.com and Allrecipes.com).\nThe recipes are structured, but they are not clean for text processing. For this reason, it was necessary to 
            preprocessed the dataset by: \n1. Filtering out incomplete examples, removing the recipes that didn't have the required
            fields (title, ingredients and instructions). \n2. Converting recipes objects into strings: RNN doesn't understand objects. 
            Therefore, it was necessary to convert recipes objects to string and then to numbers (indices). To help the RNN model to learn the structure of the text faster, 
            it was added 3 "landmarks" to it. Each section has a  unique landmark which is an emoji. \n3. Filtering out large recipes: 
            Recipes have different lengths. It was necessary to have one hard-coded sequence length limit before feeding recipe sequences to the RNN model. 
            It was found the recipe length that cover most of the recipe use-cases, and at the same time, it was set to keep it as small as possible to speed up the training process. 2,000 charachters was the picked value.
            """)
    with colp_2:
        st.image("data/recipes_book.png", use_column_width=True)

    # Define the Streamlit app
    def app():
        title = ""
        selected_ingredients=""
        # Create a text input field for the user to enter ingredients
        ingredients = st.text_input("Enter up to 5 ingredients (separated by commas)")

        # Split the input into a list of individual ingredients
        ingredient_list = [ingredient.strip() for ingredient in ingredients.split(",")]

        # Filter the recipes based on the user's input
        matching_recipes = df_full_recipe[df_full_recipe["Ingredients"].str.contains("|".join(ingredient_list))]

        # Create a selection bar with the titles of the matching recipes
        recipe_titles = matching_recipes["Title"].tolist()
        selected_title = st.selectbox("Select a recipe", [""] + recipe_titles)

        # Display the selected recipe (if any)
        if selected_title != "":
            matching_recipe_rows = matching_recipes[matching_recipes["Title"] == selected_title]
            selected_ingredients = [ingredient.strip() for ingredient in matching_recipe_rows.iloc[0]["Ingredients"].split("\n")]
            title = matching_recipe_rows.iloc[0]["Title"] # assign a value to title
            st.write(f'{matching_recipe_rows.iloc[0]["Title"]}')
            st.write('List of Ingredients:')
            st.write(f':hot_pepper:\n{matching_recipe_rows.iloc[0]["Ingredients"]}')
            st.write('Instructions:')
            st.write(f'\n{matching_recipe_rows.iloc[0]["Procedure"]}')
            # Add a download button for the selected recipe
            recipe_str = f'{matching_recipe_rows.iloc[0]["Title"]}\n\nList of Ingredients:\n{matching_recipe_rows.iloc[0]["Ingredients"]}\n\nInstructions:\n{matching_recipe_rows.iloc[0]["Procedure"]}'
            recipe_bytes = recipe_str.encode('utf-8')
            download_button = st.download_button(
                label="Download Recipe :arrow_down:",
                data=recipe_bytes,
                file_name=f"{matching_recipe_rows.iloc[0]['Title']}.txt",
            )
        elif ingredients != "":
            if not matching_recipes.empty:
                # If no recipe is selected but there are matching recipes, display the first one
                first_recipe_rows = matching_recipes.head(1)
                st.write(f'{first_recipe_rows.iloc[0]["Title"]}')
                st.write('List of Ingredients:')
                st.write(f':hot_pepper:\n{first_recipe_rows.iloc[0]["Ingredients"]}')
                st.write('Instructions:')
                st.write(f'\n{first_recipe_rows.iloc[0]["Procedure"]}')
            else:
                # If no matching recipes are found, display a message
                st.write("No matching recipes found")
        else:
            # Display instructions if no input has been provided
            st.write("#### Please enter up to 5 ingredients above")

        if selected_ingredients != None:

            # get the list of ingredients from the recipe
            selected_ingredients_user = selected_ingredients

            # Define a list of known unit of measurement keywords
            unit_keywords = ['bunch', 'bunches','clove', 'cloves', 'small', 'large', 'gram', 'g', 'kilogram', 'kg', 'milligram', 'mg', 'ounce', 'oz', 'pound', 'lb', 'tablespoon', 'tbsp', 'teaspoon', 'tsp', 'cup', 'can', 'jar', 'bottle', 'package', 'pinch', 'piece', 'slice']

            # create a dictionary to store the matched LCI names, quantity and unit of measurement
            lcis = {}
            title = title
            # iterate over the ingredients list and match each ingredient to the LCI names in the dataset
            for ingredient in selected_ingredients_user :
                ingredient = str(ingredient)
                df_extracted['LCI Name'] = df_extracted['LCI Name'].astype(str)
                # get the best match using the fuzzywuzzy library
                best_match = process.extractOne(ingredient, df_extracted['LCI Name'])
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
            empyty_nan = {'Climate change (kg CO2 eq/kg product)': [np.nan]}

            if len(selected_ingredients_user) != lcis_df.shape[0]:
                empyty_df = pd.DataFrame(columns=lcis_df.columns,data=empyty_nan)
                diff = len(selected_ingredients_user)- len(lcis_df)
                for _ in range(0, diff):
                    lcis_df=pd.concat([lcis_df, empyty_df], axis=0,ignore_index= True)

            lcis_df['original_ingredients'] = selected_ingredients_user

        
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
                'bunch': 1,
                'bunches':1
            }

            def convert_quantity(quantity, unit):
                if unit == 'egg':
                    unit = 'small'
                if isinstance(quantity, str):
                    # replace the special character with '1'
                    #quantity = quantity.replace('•', '1')
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
            
            def quantity_def(df):
                if 'Quantity' in df.columns:
                    # Remove the • character and unwanted characters like (8
                    df['Quantity'] = df['Quantity'].str.replace('•', '').str.replace(r'\(.+?\)', '').str.strip()

                    # Convert fractions to floats and handle non-numeric values
                    df['Quantity'] = df['Quantity'].apply(lambda x: eval(str(x)) if isinstance(x, str) and '/' in x else x)
                    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')

                    # Ignore NaN values
                    df = df.dropna(subset=['Quantity'])

                    df['quantity_kg'] = df.apply(lambda row: convert_quantity(row['Quantity'], row['Unit']), axis=1)
                    # Calculation of CO2 emissions for each ingredient
                    df['Environmental impact (kg CO2 eq/kg product)'] = df['quantity_kg'] * df['Climate change (kg CO2 eq/kg product)']

                    # Total CO2 emissions per recipe
                    total_impact = round(df['Environmental impact (kg CO2 eq/kg product)'].sum(),3)  # only sum non-NaN values

                    st.write('### CO2 Emissions of each of the ingredients')
                    st.write(df[['original_ingredients', 'Climate change (kg CO2 eq/kg product)','Environmental impact (kg CO2 eq/kg product)']])

                    # Display total CO2 emissions
                    st.metric('Total CO2 Emissions', f'{total_impact} (kg CO2 eq/kg product)')

                    # Total single EF per recipe
                    total_single_ef = sum(row['Single EF 3.1 score (mPt/kg product)'] * row['quantity_kg']
                                            for index, row in df.iterrows())
                    # Display total CO2 emissions
                    st.metric('Total Single EF 3.1 score', f'{round(total_single_ef,3)} (mPt/kg product)')
                else:
                    print('No quantity')
                return df

            quantity_def(lcis_df)


    if __name__ == '__main__':
        app()

    st.write('Source of the recipes dataset: [link](https://eightportions.com/datasets/Recipes/)')


########################################### Environmental footprint of the products

if selected2 == "Environmental footprint of products dataset":
    st.markdown("## Environmental footprint of products dataset")

    st.markdown("---")


    # Sidebar 
    st.sidebar.markdown(
        "**Note:** The following selection interacts with the Table 1 and the Fig.1."
    )
    st.sidebar.markdown("## Selection")


    st.markdown("#### Complete dataset")

    # Infornation about the entire dataset
    with st.expander("Check the complete dataset for the environmental footprint of more than 2,000 food products"):

        # Set up the grid options
        gb = GridOptionsBuilder.from_dataframe(df_products)
        gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=True)
        gridOptions = gb.build()

        # Add the filter to the "Food Group" column
        gridOptions['columnDefs'][0]['filter'] = True

        # Display the AG Grid with the filter
        AgGrid(df_products, gridOptions=gridOptions, height=400, width='100%')


    subgroup_emissions = df_products.groupby(by=['Food Group','Food Subgroup']).sum()[["Climate change (kg CO2 eq/kg product)"]]

    fig_2 = px.scatter(subgroup_emissions.reset_index(), 
                 x='Climate change (kg CO2 eq/kg product)', 
                 y='Food Subgroup', 
                 size='Climate change (kg CO2 eq/kg product)', 
                 color='Food Group',
                 hover_data=['Climate change (kg CO2 eq/kg product)'],
                 title='Fig.2: CO2 Emissions (kg CO2 eq/kg product) by Food Group and Food Subgroup')
    st.plotly_chart(fig_2)
    fig_2.write_html('data.html')

    # Interaction with the sidebar selection
    # Sidebar selection
    selected_food_group = st.sidebar.selectbox('Select Food Group', df_products['Food Group'].unique())
    selected_column = st.sidebar.selectbox('Select Column', ['Single EF 3.1 score (mPt/kg product)', 
                                                            'Climate change (kg CO2 eq/kg product)', 
                                                            'Water resource depletion (m3 depriv./kg product)', 
                                                            'Toxicological effects on human health: non-carcinogens (CTUh/kg product)', 
                                                            'Toxicological effects on human health: carcinogenic substances (CTUh/kg product)', 
                                                            'Land use (Pt/kg product)'])

    # Filter data by selected food group
    df_filtered = df_products[df_products['Food Group'] == selected_food_group]
    # Create plot
    fig = px.bar(df_filtered, x='Food Subgroup', y=selected_column, color='Packaging Material', 
                title='Fig.1: {} by Product for {}'.format(selected_column, selected_food_group))
    fig.update_xaxes(title='Product')
    fig.update_yaxes(title=selected_column)



    # Display plot
    st.plotly_chart(fig, use_container_width=True)
    # Display table
    st.write('Table 1: Selected Food Group: {}'.format(selected_food_group))
    st.write(df_filtered)


    st.markdown("#### Greenhouse gas emissions across the food products supply chain")
    st.markdown("""
    You can find a structured analysis of the global emissions across the food chain by clicking the following button :bar_chart:
    """)
    url = 'http://127.0.0.1:8050'
    if st.button('Open Dashboard'):
        webbrowser.open_new_tab(url)







