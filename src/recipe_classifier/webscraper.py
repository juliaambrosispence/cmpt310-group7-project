# webscraper.py
# Adds functionality to scrape recipe data from a website

from bs4 import BeautifulSoup
from urllib.request import urlopen
import json
from playwright.sync_api import sync_playwright
import re

keep_keywords = [
    'Asian',
    'Mexican', 'Canadian',
    'Southwestern U.S.',
    'Australian', 'Indian', 'African', 'Chinese', 'Southwest Asia (middle East)',
    'Greek', 'Caribbean', 'South American', 'Cajun', 'German',
    'Scandinavian', 'Creole', 'Spanish', 'Thai', 'Moroccan', 'Japanese', 'Scottish', 'Portuguese', 'New Zealand',
    'Hawaiian', 'Swiss', 'Korean', 'Lebanese', 'South African', 'Hungarian', 'Russian', 'Vietnamese', 'Welsh',
    'Swedish', 'Brazilian', 'Austrian',
    'Turkish', 'Indonesian', 'Norwegian', 'Peruvian', 'Native American', 'Polynesian', 'Dutch', 'Polish', 'Danish',
    'Belgian', 'Szechuan', 'Pennsylvania Dutch', 'Czech', 'Egyptian', 'Cuban', 'Finnish', 'Filipino', 'Malaysian',
    'Venezuelan',
    'Guatemalan', 'Nigerian', 'Colombian', 'Palestinian', 'Puerto Rican', 'Ethiopian', 'Iraqi', 'Cantonese',
    'Cambodian', 'Hunan', 'Chilean', 'Pakistani', 'Icelandic', 'Costa Rican', 'Nepalese', 'Sudanese', 'Honduran',
    'Ecuadorean'

]

def parse_url(url):

    data = {}

    #Launch playwright headless browser to extract ingredients
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url)

        # Print title of page to scrape data
        print(f"Extracting recipe from page {page.title()}")

        # Run Javascript commands on metadata manager to get ingredient info
        data["ingredients"] = page.evaluate("""() => window.mdManager.getParameter("ingredients", ", ")""")
        ingredients_list = [i.strip() for i in data["ingredients"].split(",") if i.strip()]
        data["ingredients"] = ingredients_list

        browser.close()

    # Parse HTML to find JSON data in page

    page = urlopen(url)
    html = page.read().decode("utf-8")
    soup = BeautifulSoup(html, "html.parser")

    # Extract JSON data from webpage
    json_extract = soup.find("script", attrs={"type": "application/ld+json"})
    #print(json_extract)

    data_raw = json.loads(json_extract.text)

    #print(data_raw)

    # Clean up data

    # Convert times
    match = re.match(r"^PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?$", data_raw["cookTime"])
    if match:
        hours = match.group(1) or 0
        minutes = match.group(2) or 0
        seconds = match.group(3) or 0
        data["cook_time"] = int(hours)*60 + int(minutes) + int(seconds)/60
    else:
        data["cook_time"] = 0

        match = re.match(r"^PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?$", data_raw["prepTime"])
    if match:
        hours = match.group(1) or 0
        minutes = match.group(2) or 0
        seconds = match.group(3) or 0
        data["prep_time"] = int(hours)*60 + int(minutes) + int(seconds)/60
    else:
        data["prep_time"] = 0

    # Try and find keyword, otherwise None
    data["keyword"] = "None"
    keywords_list = [i.strip() for i in data_raw["keywords"].split(",") if i.strip()]
    for keyword in keywords_list:
        if keyword in keep_keywords:
            data["keyword"] = keyword

    # Extract other data
    data["calories"] = data_raw["nutrition"]["calories"]
    data["fat"] = data_raw["nutrition"]["fatContent"]
    data["sat_fat"] = data_raw["nutrition"]["saturatedFatContent"]
    data["cholesterol"] = data_raw["nutrition"]["cholesterolContent"]
    data["sodium"] = data_raw["nutrition"]["sodiumContent"]
    data["carbs"] = data_raw["nutrition"]["carbohydrateContent"]
    data["fiber"] = data_raw["nutrition"]["fiberContent"]
    data["sugar"] = data_raw["nutrition"]["sugarContent"]
    data["protein"] = data_raw["nutrition"]["proteinContent"]

    raw_servings = data_raw["recipeYield"];
    match = re.search(r'(\d+)\s*-\s*(\d+)', raw_servings);#look for ranges first
    if match:
        min = int(match.group(1));
        max = int(match.group(2));
        data["servings"] = (min + max) // 2;  #just average the range for full integer 
    else:
        digits_only = re.sub(r'\D', '', raw_servings);
        data["servings"] = int(digits_only) if digits_only else 2;  #default
    #print(data)

    return data

