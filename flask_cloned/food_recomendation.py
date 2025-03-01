
# ['Weight loss', 'Muscle gain', 'General health', 'Indulgence']

# l=['Protein-Rich Foods', 'Whole Grains', 'Vegetables', 'Dairy', 'Fruits', 'Vegetarian', 'Nuts', 'Spices', 'Fats', 'Sweets', 'Beverages', 'Seafood', 'Snacks', 'Condiments', 'Street Food', 'Indian', 'South Indian', 'Fusion', 'North Indian', 'Tibetan', 'Indo-Chinese', 'Gujarati', 'Rajasthani']

# ['Lactose intolerant', 'No Restriction', 'Gluten-free', 'Vegetarian', 'Caffeine sensitive']


import pandas as pd

weight = float(input("Enter your weight (kg):"))
height = float(input("Enter your height (cm):"))
gender = input("Gender:")

# activity_level_options = ['Sedentary', 'Lightly_Active', 'Moderately_Active', 'Very_Active']
activity_level = input(["['Sedentary', 'Lightly_Active', 'Moderately_Active', 'Very_Active']"])

age = float(input("Select your age:"))
bmi = round(weight/((height/100)**2),2)

if bmi < 18.5:
    bodytype="Underweight"
elif 18.5 <= bmi < 25:
    bodytype= "Normal Weight"
elif 25 <= bmi < 30:
    bodytype= "Overweight"
else:
    bodytype= "Obese"
    
print(bodytype)

if gender.lower()=="male" or gender=="m":
    bmr=88.362+(13.397*weight)+(4.799*height)-(5.677*age)
elif gender.lower()=="Female" or gender=="f":
    bmr=447.593+(9.247*weight)+(3.098*height)-(4.330*age)
else:
    print("Fuck Off")
    
print(bmr)


multiplier = {
    'sedentary': 1.2,
    'lightly_active': 1.375,
    'moderately_active': 1.55,
    'very_active': 1.725
    }

tdee = bmr*multiplier[activity_level.lower()]  
print("TDEE : ",tdee)



def check_match(value, items):
    if pd.isna(value):
        return False
    for item in items:
        if item in value:
            return True
    return False

def recommend_food(goals, preferences, restrictions, filtered_df):
    goal_weight = 1
    preference_weight = 2
    restriction_weight = -3
    
    filtered_df.loc[:, 'Score'] = (
        filtered_df['Goals'].apply(lambda x: goal_weight if x in goals else 0) +
        filtered_df['Preferences'].apply(lambda x: preference_weight if check_match(x, preferences) else 0) +
        filtered_df['Restrictions'].apply(lambda x: restriction_weight if check_match(x, restrictions) else 0)
    )

    recommended_df = filtered_df.sort_values(by='Score', ascending=False)  # Recommend top 6 foods
    
    return recommended_df

goals = ['Muscle gain']
preferences = ['Protein-Rich Foods', 'Whole Grains']
restrictions = ['No Restriction']

filtered_df = data[data['Goals'].isin(goals) & data['Preferences'].apply(lambda x: check_match(x, preferences)) & data['Restrictions'].apply(lambda x: check_match(x, restrictions))]

if not filtered_df.empty:
    recommendations = recommend_food(goals, preferences, restrictions, filtered_df)
    print("Recommended Foods:")
    print(recommendations[['food', 'calorie', 'protein', 'carbs', 'fat']])
else:
    print("No recommendations available for the given inputs.")

