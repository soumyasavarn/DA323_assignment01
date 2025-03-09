import os
import json
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://nationalanthems.info/"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36'
}

def load_countries(filename):
    """Load the countries JSON file."""
    with open(filename, 'r', encoding='utf-8') as f:
        countries = json.load(f)
    return countries

def select_countries(countries, n=255):
    """Select n countries from the provided dictionary.
       Here, we choose the first n keys in alphabetical order.
    """
    keys = list(countries.keys())
    keys.sort()  # sort alphabetically by country code
    selected = keys[254:n]
    return {code: countries[code] for code in selected}

def scrape_country_page(code):
    """Scrape a given country's page for the English anthem translation and MP3 link.
    
    The URL is assumed to be BASE_URL + code.lower() + ".htm"
    """
    url = BASE_URL + code.lower() + ".htm"
    try:
        response = requests.get(url, headers=HEADERS)
    except Exception as e:
        print(f"Error fetching URL {url}: {e}")
        return None, None
    
    if response.status_code != 200:
        print(f"Page not found or blocked for {code} at URL: {url} - Response: {response}")
        return None, None
    
    soup = BeautifulSoup(response.text, 'html.parser')
    # print (soup)
    # --- Extract the English Translation ---
    translation = None
    header_div = soup.find("div", class_="collapseomatic", title=lambda t: t in ["English translation", "English lyrics"])

    # print (header_div)
    if header_div:
    # Construct the corresponding content div's id
        content_id = "target-" + header_div["id"]
        content_div = soup.find("div", id=content_id)
        if content_div:
        # Extract and clean the text, using a space separator for any <br/> tags
            translation_text = content_div.get_text(separator=" ", strip=True)
            # print("Extracted Translation:")
            translation = (translation_text)
        else:
            print("Content div not found.")
    else:
        print("Header div with title 'English translation' not found.")

    # --- Extract the MP3 Link ---
    mp3_link = BASE_URL + code.lower() + ".mp3"
    print (mp3_link)

    
    # for a in soup.find_all('a', href=True):
    #     href = a['href']
    #     print (href)
    #     if href.lower().endswith(".mp3"):
    #         # Ensure the link is complete (absolute URL)
    #         if not href.startswith("http"):
    #             mp3_link = BASE_URL + href
    #         else:
    #             mp3_link = href
    #         break

    return translation, mp3_link

def save_text(content, filepath):
    """Save text content to a file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

def download_file(url, filepath):
    """Download file from the given URL and save to the filepath."""
    try:
        response = requests.get(url, headers=HEADERS, stream=True)
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
        else:
            print(f"Failed to download file from {url} - Response: {response}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

def main():
    # Load the countries list from the provided JSON file
    countries = load_countries("countries.json")
    
    # Select 105 countries from the list (5 extra for failure cases compensatation)
    selected_countries = select_countries(countries)
    print (selected_countries)
    
    print(f"Selected {len(selected_countries)} countries for processing.")
    
    # Create directories for translations and mp3 files
    os.makedirs("translations", exist_ok=True)
    os.makedirs("mp3", exist_ok=True)
    
    for code, name in selected_countries.items():
        print(f"\nProcessing {code} - {name}")
        translation, mp3_link = scrape_country_page(code)
        
        # Save the translation if found
        if translation:
            safe_name = name.replace(" ", "_")
            translation_filepath = os.path.join("translations", f"{code}_{safe_name}.txt")
            save_text(translation, translation_filepath)
            print(f"Saved translation to {translation_filepath}")
        else:
            print(f"No English translation found for {name}.")
        
        # Download the MP3 file if available
        
        if mp3_link:
            safe_name = name.replace(" ", "_")
            mp3_filepath = os.path.join("mp3", f"{code}_{safe_name}.mp3")
            download_file(mp3_link, mp3_filepath)
            print(f"Downloaded MP3 to {mp3_filepath}")
        else:
            print(f"No MP3 file found for {name}.")

if __name__ == "__main__":
    main()
