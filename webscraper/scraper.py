### Can scrape a page nicely, debug the scrape_text_from_url function for iterative process

import os
import requests
from bs4 import BeautifulSoup

def get_page_urls(base_url, total_pages):
    page_urls = []
    for i in range(1, total_pages + 1):
        page_urls.append(f"{base_url}/page/{i}")
        print("Successful!", page_urls)
    return page_urls

def scrape_text_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')

        text = soup.get_text()
        print("Read the text!")
        return text
    return None

def save_text_to_file(text, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(text)

def main(base_url, total_pages, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    page_urls = get_page_urls(base_url, total_pages)
    
    for i, url in enumerate(page_urls):
        text = scrape_text_from_url(url)
        if text:
            filename = os.path.join(output_dir, f"name_of_the_file.txt")
            save_text_to_file(text, filename)
            print(f"Saved: {filename}")
        else:
            print(f"Failed to scrape: {url}")

base_url = "URL_OF_ANY_SCRAPEABLE_WEBSITE"  # URL of the webpage to be scraped
total_pages = 1 
output_dir = "output_directory_path" # Replace the path of output directory

main(base_url, total_pages, output_dir)
