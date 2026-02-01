import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import random
import csv
import re

# Progress record file
PROGRESS_FILE = 'crawl_progress.txt'

# ==========================================================
# Functions to manage crawling progress (save and load)
# ==========================================================
def save_progress(category_url, current_page):
    """Save current crawling progress"""
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        f.write(f"{category_url}|{current_page}")

def load_progress():
    """Load previous crawling progress"""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content:
                url, page = content.split('|')
                return url, int(page)
    return None, 1  # Default to page 1

# ==========================================================
# Function to persist scraped shop data into a CSV file
# ==========================================================
def save_csv(data):
    file_path = "urls_output.csv"
    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(['Shop URL', 'Subcategory ID'])
        for item in data:
            writer.writerow(item)
    print(f"Data successfully written to {file_path}!")

# ==========================================================
# Main logic to crawl shop lists from a specific category URL
# ==========================================================
def crawl_category(category_url, start_page=1):
    driver = webdriver.Chrome()
    try:
        driver.get(category_url)
        driver.maximize_window()
        time.sleep(50)  # Wait for manual login

        current_page = start_page

        while True:
            print(f"Crawling page {current_page}...")

            # Simulate scrolling
            for _ in range(6):
                driver.execute_script("window.scrollBy(0, 300);")
                time.sleep(random.uniform(0.1, 2))

            # Parse page
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            shop_list_div = soup.find('div', id='shop-all-list')

            if not shop_list_div:
                print("Container id='shop-all-list' not found")
                break

            shop_data = []
            for li in shop_list_div.find_all('li'):
                pic_div = li.find('div', class_='pic')
                shop_id = pic_div.find('a')['data-shopid'] if pic_div and pic_div.find('a') else None

                tag_addr_div = li.find('div', class_='tag-addr')
                subcategory_id = None
                if tag_addr_div:
                    a_tag = tag_addr_div.find('a', href=True)
                    if a_tag:
                        match = re.search(r'/g(\d+)', a_tag['href'])
                        subcategory_id = match.group(1) if match else None

                if shop_id and subcategory_id:
                    shop_url = f'https://www.dianping.com/shop/{shop_id}'
                    shop_data.append((shop_url, subcategory_id))

            if shop_data:
                save_csv(shop_data)

            # Save current progress
            save_progress(category_url, current_page)

            # Check for next page button
            next_buttons = driver.find_elements(By.XPATH, '//div[@class="page"]//a[@class="next" and @title="下一页"]')
            if not next_buttons:
                print("No more pages, crawling finished")
                break

            # Navigate to next page
            sleep_time = random.randint(1, 10)
            time.sleep(sleep_time)
            next_buttons[0].click()
            time.sleep(sleep_time)

            current_page += 1

    except Exception as e:
        print(f"Exception during crawling: {str(e)}")
        print(f"Progress saved: {category_url} Page {current_page}")
    finally:
        driver.quit()

# List of category URLs
category_urls = [
    'https://www.dianping.com/shanghai/ch10/g101',  # Jiangzhe Cuisine
    'https://www.dianping.com/shanghai/ch10/g113',  # Japanese Cuisine
    'https://www.dianping.com/shanghai/ch10/g116',  # Western Food
    'https://www.dianping.com/shanghai/ch10/g103',  # Cantonese Cuisine
    'https://www.dianping.com/shanghai/ch10/g114',  # Korean Cuisine
    'https://www.dianping.com/shanghai/ch10/g102',  # Sichuan Cuisine
    'https://www.dianping.com/shanghai/ch10/g115',  # Southeast Asian Cuisine
    'https://www.dianping.com/shanghai/ch10/g106',  # Northeast Chinese Cuisine
    'https://www.dianping.com/shanghai/ch10/g3243', # Xinjiang Cuisine
    'https://www.dianping.com/shanghai/ch10/g104',  # Hunan Cuisine
    'https://www.dianping.com/shanghai/ch10/g311',  # Beijing Cuisine
    'https://www.dianping.com/shanghai/ch10/g34351', # Local Cuisines
    'https://www.dianping.com/shanghai/ch10/g234',  # Middle Eastern Cuisine
    'https://www.dianping.com/shanghai/ch10/g2797', # African Cuisine
    'https://www.dianping.com/shanghai/ch10/g34236', # Drinks
    'https://www.dianping.com/hefei/ch10/g26482',   # Anhui Cuisine
    'https://www.dianping.com/qingdao/ch10/g26483'  # Shandong Cuisine
]

# Load previous progress and start crawling
last_url, last_page = load_progress()

start_crawling = False
for url in category_urls:
    if not start_crawling:
        if last_url is None or url == last_url:
            start_crawling = True
            crawl_category(url, last_page)
    else:
        crawl_category(url)