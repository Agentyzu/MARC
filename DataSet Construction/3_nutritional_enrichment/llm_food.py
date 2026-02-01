import csv
import json
import time
import os
import requests
from openai import OpenAI
from promptTemplate import mealPrompt

# Initialize OpenAI client
client = OpenAI(api_key="", base_url="https://api.deepseek.com")

# USDA API key
USDA_API_KEY = ""

# ==========================================================
# Translation logic to convert ingredient names for API compatibility
# ==========================================================
def translate_to_english(chinese_name):
    """Translate Chinese ingredient names to English via LLM"""
    print(f"\n[Translation] Translating ingredient: {chinese_name}")
    system_prompt = "You are a professional translation assistant. Accurately translate Chinese ingredient names to English."
    user_prompt = f"Please translate the following Chinese ingredient name to English. Return only the English name without explanation: '{chinese_name}'"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.1
        )
        english_name = response.choices[0].message.content.strip()
        print(f"[Result] {chinese_name} → {english_name}")
        return english_name
    except Exception as e:
        print(f"[Error] Error translating '{chinese_name}': {str(e)}")
        return None

# ==========================================================
# USDA API integration for standard nutritional data lookup
# ==========================================================
def query_usda_nutrition(english_name):
    """Query USDA API for nutritional data, defaulting missing fields to 0"""
    print(f"\n[USDA Query] Querying: {english_name}")
    try:
        encoded_name = requests.utils.quote(english_name)
        url = f"https://api.nal.usda.gov/fdc/v1/foods/search?api_key={USDA_API_KEY}&query={encoded_name}&dataType=Foundation,SR%20Legacy&sortBy=dataType.keyword"
        
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        nutrition_data = {'energy': 0, 'carbs': 0, 'protein': 0, 'fat': 0}
        
        if not data.get('foods'):
            return nutrition_data
            
        # Prioritize Foundation data, then SR Legacy
        all_foods = [f for f in data['foods'] if f.get('dataType') == 'Foundation'] + \
                    [f for f in data['foods'] if f.get('dataType') == 'SR Legacy']
        
        for food in all_foods:
            nutrients = food.get('foodNutrients', [])
            temp_nutrition = {'energy': 0, 'carbs': 0, 'protein': 0, 'fat': 0}
            
            for nutrient in nutrients:
                n_name = nutrient.get('nutrientName', '').lower()
                val = nutrient.get('value', 0)
                unit = nutrient.get('unitName', '').lower()
                
                if 'energy' in n_name and unit == 'kcal':
                    temp_nutrition['energy'] = val
                elif 'carbohydrate' in n_name and unit == 'g':
                    temp_nutrition['carbs'] = val
                elif 'protein' in n_name and unit == 'g':
                    temp_nutrition['protein'] = val
                elif 'fat' in n_name and unit == 'g' and 'total' in n_name:
                    temp_nutrition['fat'] = val
            
            # Use the first record that contains energy data
            if temp_nutrition['energy'] > 0:
                nutrition_data = temp_nutrition
                break
        
        return nutrition_data
    except Exception as e:
        print(f"[USDA Error] Error querying {english_name}: {str(e)}")
        return {'energy': 0, 'carbs': 0, 'protein': 0, 'fat': 0}

# ==========================================================
# Fallback LLM logic for estimating unknown nutritional values
# ==========================================================
def query_calorie_from_llm(ingredient_name):
    """Estimate calories via LLM when USDA data is unavailable"""
    print(f"\n[LLM Query] Querying calories: {ingredient_name}")
    system_prompt = "You are a professional nutritionist. Provide calorie info based on your knowledge."
    user_prompt = f"Tell me the calories per 100g for '{ingredient_name}'. Return only a number, no units or explanations."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.3
        )
        return float(response.choices[0].message.content.strip())
    except Exception as e:
        return 0

# ==========================================================
# Core logic to decompose dishes and calculate total nutrients
# ==========================================================
def meal_detail_by_LLM(food: str):
    """Fetch dish details and ingredients using LLM, then calculate cumulative nutrition"""
    print(f"\n[Processing] Starting: {food}")
    
    # 1. Ask LLM for recipe composition (ingredients and weight)
    messages = [
        {"role": "system", "content": mealPrompt},
        {"role": "user", "content": f"Analyze the following dish: {food}, identify main ingredients and weights."}
    ]
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        response_format={'type': 'json_object'}
    )
    meal_data = json.loads(response.choices[0].message.content)
    ingredients = meal_data.get("ingredient", [])
    
    total_calorie = total_carbs = total_protein = total_fat = 0.0
    gram_explanations, calorie_explanations = [], []
    
    # 2. Iterate through ingredients to gather nutritional data
    for item in ingredients:
        name = item.get("name", "").strip()
        grams = item.get("grams", 0)
        gram_explanations.append(f"{name}:{grams}g")
        
        eng_name = translate_to_english(name)
        nutrition = query_usda_nutrition(eng_name) if eng_name else {'energy':0,'carbs':0,'protein':0,'fat':0}
        
        cal_per_100g = nutrition['energy']
        source = "(USDA)" if cal_per_100g != 0 else "(LLM Estimate)"
        
        if cal_per_100g == 0:
            cal_per_100g = query_calorie_from_llm(name)
            
        total_calorie += (grams / 100) * cal_per_100g
        total_carbs += (grams / 100) * nutrition['carbs']
        total_protein += (grams / 100) * nutrition['protein']
        total_fat += (grams / 100) * nutrition['fat']
        
        calorie_explanations.append(f"{name}{grams}g{source}({cal_per_100g}kcal/100g) → {(grams/100)*cal_per_100g:.1f}kcal")
    
    return {
        "菜名": food,
        "type": meal_data.get("type", ""),
        "ingredient": ",".join([i["name"] for i in ingredients]),
        "主要食材克数": "；".join(gram_explanations),
        "calorie": round(total_calorie, 1),
        "explained": "；".join(calorie_explanations) + f"；Total: {total_calorie:.1f}kcal",
        "carbs": round(total_carbs, 1),
        "protein": round(total_protein, 1),
        "fat": round(total_fat, 1),
        "s": "，".join(meal_data.get("s", [])),
        "food_tag": "，".join(meal_data.get("food_tag", [])),
        "推荐人数": "", "图片地址": "", "子分类ID": "", "父分类ID": "", "分类名称": ""
    }

# ==========================================================
# Batch processing management for CSV updates
# ==========================================================
def update_csv_with_meal_details(input_csv_path, output_csv_path):
    """Read input CSV and write enriched data to output, supporting progress resumption"""
    fieldnames = [
        "菜名", "type", "ingredient", "主要食材克数", "calorie", "explained",
        "carbs", "protein", "fat", "s", "food_tag", "推荐人数", "图片地址", 
        "子分类ID", "父分类ID", "分类名称"
    ]

    if not os.path.exists(output_csv_path):
        with open(output_csv_path, mode='w', newline='', encoding='UTF-8-SIG') as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    processed_foods = set()
    try:
        with open(output_csv_path, mode='r', encoding='UTF-8-SIG') as f:
            for row in csv.DictReader(f):
                if row["菜名"]: processed_foods.add(row["菜名"].strip())
    except: pass

    with open(input_csv_path, mode='r', encoding='UTF-8-SIG') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            food_name = row["菜名"].strip()
            if not food_name or food_name in processed_foods:
                continue

            try:
                llm_data = meal_detail_by_LLM(food_name)
                merged_row = {**llm_data, "推荐人数": row.get("推荐人数", ""), "图片地址": row.get("图片地址", ""),
                              "子分类ID": row.get("子分类ID", ""), "父分类ID": row.get("父分类ID", ""), "分类名称": row.get("分类名称", "")}

                with open(output_csv_path, mode='a', newline='', encoding='UTF-8-SIG') as outfile:
                    csv.DictWriter(outfile, fieldnames=fieldnames).writerow(merged_row)
                time.sleep(1)
            except Exception as e:
                print(f"Failed: {food_name} - {str(e)}")

if __name__ == '__main__':
    update_csv_with_meal_details('clean_name.csv', 'llm_food_data.csv')