from flask import Flask, render_template, request, redirect, url_for, jsonify
import pandas as pd
from fuzzywuzzy import process
from flask_cors import CORS


app = Flask(__name__)
CORS(app)


tracker = pd.read_csv("tracker.csv")


def match_food(x, df):
    match_food, similar_score = process.extractOne(x.capitalize(), df['food'].str.capitalize().values)
    try:
        if similar_score >= 80:
            row = df.loc[df['food'].str.capitalize() == match_food.capitalize()]
            calorie = float(row['calorie'].values[0])
            protein = float(row['protein'].values[0])
            carbs = float(row['carbs'].values[0])
            fat = float(row['fat'].values[0])
            sugar = float(row['Sugar'].values[0])
            fiber = float(row['Fiber'].values[0])
            sodium = float(row['Sodium'].values[0])
            vitamin = float(row['Vitamins'].values[0])
            mineral = float(row['Minerals'].values[0])
            return (calorie, protein, carbs, fat, sugar, fiber, sodium, vitamin, mineral, match_food)
    except Exception as e:
        print("Error:", e)
        return None


@app.route('/calorie_tracker', methods=['POST'])
def convert_code():
    
    food_dict={}

    if request.method == 'POST':
        x = request.json.get('food')
        serv = request.json.get('serving')
        print("Food : ",x)
        print("Serving : ",serv)
        ma = match_food(x, tracker)

        if ma:
            food_dict = {
                "Food": ma[9],
                "Serving": serv,
                "Calorie": ma[0] * serv,
                "Protein": ma[1] * serv,
                "Carbs": ma[2] * serv,
                "Fat": ma[3] * serv,
                "Sugar": ma[4] * serv,
                "Fiber": ma[5] * serv,
                "Sodium": ma[6] * serv,
                "Vitamins": ma[7] * serv,
                "Minerals": ma[8] * serv
            }
            return jsonify({'Calorie': food_dict})
        else:
            return "Food not found!"
        
    return jsonify({'Calorie': food_dict})


if __name__ == '__main__':
    app.run(debug=True, port=8001) 






