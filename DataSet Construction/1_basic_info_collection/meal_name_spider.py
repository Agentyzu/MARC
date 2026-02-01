import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import random
import csv

# Status file path
STATUS_FILE = "crawler_status_name.txt"

# ==========================================================
# Functions to handle crawling state to support resuming
# ==========================================================
def load_status():
    """Load the state from the last interruption"""
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE, 'r') as f:
            return int(f.read().strip())
    return 0

def save_status(index):
    """Save the index of the currently processed URL"""
    with open(STATUS_FILE, 'w') as f:
        f.write(str(index))

# ==========================================================
# Data loading and CSV writing utilities
# ==========================================================
def load_url(file_path):
    urls = []
    subcategory_ids = []
    with open(file_path, "r", newline="", encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if row:
                urls.append(row[0])
                subcategory_ids.append(row[1])
    print(f"Total loaded: {len(urls)} URLs")
    return urls, subcategory_ids

def save_csv(file_path, rows, write_header=False):
    file_exists = os.path.exists(file_path)
    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if write_header and not file_exists:
            writer.writerow(['Dish Name', 'Recommendation Count', 'Image URL', 'Subcategory ID'])
        writer.writerows(rows)
    print(f"Data successfully written to {file_path}!")

# ==========================================================
# Main crawler logic for extracting dish details
# ==========================================================
def get_meals(urls, subcategory_ids, output_path):
    options = webdriver.ChromeOptions()
    options.add_argument(
        'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)

    # Load status to resume
    start_index = load_status()
    print(f"Resuming crawl from URL index {start_index}")

    try:
        driver = webdriver.Chrome(options=options)
        # Initialize from the last position
        url = urls[start_index]
        driver.get(url)
        driver.maximize_window()
        time.sleep(50)  # Wait for manual login

        for i in range(start_index, len(urls)):
            try:
                if i % 10 == 0:
                    time.sleep(random.randint(15, 20))
                sleep_time1 = random.randint(4, 10)
                sleep_time2 = random.randint(4, 10)

                current_url = urls[i]
                current_subcategory_id = subcategory_ids[i]
                driver.get(current_url)
                time.sleep(sleep_time1)

                soup = BeautifulSoup(driver.page_source, 'html.parser')

                # Simulate human-like scrolling
                for k in range(0, 2):
                    driver.execute_script(f"window.scrollBy(0, 300);")
                    time.sleep(random.uniform(0.1, 2))

                dish_divs = soup.find('div', class_='dishNameContainer wx-view')

                dish_name = []
                dish_recommend = []
                dish_images = []

                # Extract dish names
                if dish_divs:
                    for wx_view in dish_divs.find_all('div', class_='wx-view'):
                        div = wx_view.find('div', class_='dishName wx-view')
                        if div:
                            dish_name.append(div.text.strip())
                        else:
                            continue

                # Extract recommendation info
                recommend_divs = soup.find_all('div', class_='recommendInfo wx-view')
                if recommend_divs:
                    for recommend_div in recommend_divs:
                        span = recommend_div.find('span', class_='recomment-text wx-text')
                        if span:
                            dish_recommend.append(span.text.strip())
                        else:
                            continue

                # Extract image URLs
                image_divs = soup.find_all('div', class_='lazyload-image')
                if image_divs:
                    for image_div in image_divs:
                        style = image_div.get('style', '')
                        image_url = style.split('url("')[-1].split('")')[0] if 'url("' in style else ''
                        dish_images.append(image_url)

                print(dish_name)
                print(dish_recommend)
                print(dish_images)
                print(f"Current Index: {i}")

                # Organize data for saving
                data_to_save = []
                for idx, name in enumerate(dish_name):
                    recommend = dish_recommend[idx] if idx < len(dish_recommend) else "No recommendation count"
                    image = dish_images[idx] if idx < len(dish_images) else "No image URL"
                    data_to_save.append([name, recommend, image, current_subcategory_id])

                if i == 0:
                    save_csv(output_path, data_to_save, write_header=True)
                else:
                    save_csv(output_path, data_to_save)

                # Update status
                save_status(i + 1)
                time.sleep(sleep_time2)

            except Exception as e:
                print(f"Error processing URL {i}: {str(e)}")
                save_status(i)
                continue

    except Exception as e:
        print(f"Critical crawler error: {str(e)}")
    finally:
        if 'driver' in locals():
            driver.quit()
        print("Crawler stopped")

if __name__ == '__main__':
    url_path = "unique_urls.csv"
    output_path = "name_output.csv"
    urls, subcategory_ids = load_url(url_path)
    get_meals(urls, subcategory_ids, output_path)