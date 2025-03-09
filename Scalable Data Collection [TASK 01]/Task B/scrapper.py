import os
import re
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
import shutil
from urllib.parse import urljoin

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

categories = {
    "Technology": [
        "https://techcrunch.com",
        "https://thenextweb.com",
        "https://www.wired.com"
    ],
    "Health": [
        "https://www.webmd.com",
        "https://www.health.com",
        "https://www.medicalnewstoday.com"
    ],
    "Finance": [
        "https://www.forbes.com/finance",
        "https://www.bloomberg.com/markets",
        "https://www.cnbc.com/finance"
    ],
    "Sports": [
        "https://www.espn.com",
        "https://www.si.com",
        "https://sports.yahoo.com"
    ],
    "Entertainment": [
        "https://www.tmz.com",
        "https://www.etonline.com",
        "https://www.hollywoodreporter.com"
    ],
    "Travel": [
        "https://www.lonelyplanet.com",
        "https://www.travelandleisure.com",
        "https://www.nationalgeographic.com/travel"
    ],
    "Education": [
        "https://www.edutopia.org",
        "https://www.educationworld.com",
        "https://www.timeshighereducation.com"
    ],
    "Politics": [
        "https://www.politico.com",
        "https://www.cnn.com/politics",
        "https://www.nytimes.com/section/politics"
    ],
    "Science": [
        "https://www.sciencemag.org",
        "https://www.nature.com",
        "https://www.sciencenews.org"
    ],
    "Food": [
        "https://www.foodnetwork.com",
        "https://www.epicurious.com",
        "https://www.seriouseats.com"
    ],
    "Fashion": [
        "https://www.vogue.com",
        "https://www.elle.com",
        "https://www.gq.com"
    ],
    "Art": [
        "https://www.artnews.com",
        "https://www.artsy.net",
        "https://www.mutualart.com"
    ],
    "Business": [
        "https://www.businessinsider.com",
        "https://www.ft.com",
        "https://www.wsj.com"
    ],
    "Environment": [
        "https://www.ecowatch.com",
        "https://www.nationalgeographic.com/environment",
        "https://www.greenpeace.org"
    ],
    "Automotive": [
        "https://www.caranddriver.com",
        "https://www.topgear.com",
        "https://www.motortrend.com"
    ],
    "Real_Estate": [
        "https://www.zillow.com/blog",
        "https://www.realtor.com/news",
        "https://www.curbed.com"
    ],
    "Culture": [
        "https://www.bbc.com/culture",
        "https://www.newyorker.com/culture",
        "https://www.theguardian.com/culture"
    ],
    "Law": [
        "https://www.law.com",
        "https://www.legalcheek.com",
        "https://www.abajournal.com"
    ],
    "History": [
        "https://www.history.com",
        "https://www.smithsonianmag.com/history",
        "https://www.ancient-origins.net"
    ],
    "Gaming": [
        "https://www.ign.com",
        "https://www.gamespot.com",
        "https://www.polygon.com"
    ]
}

output_dir = "TextWorldCorpus"
os.makedirs(output_dir, exist_ok=True)

def clean_text(text):
    # Remove HTML tags (if any remain)
    text = re.sub(r'<[^>]+>', '', text)
    # Remove punctuation and non-alphanumeric characters
    text = re.sub(r'[^\w\s]', '', text)
    # Lowercase the text
    text = text.lower()
    # Remove stop words
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def crawl_page(url):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return None
        soup = BeautifulSoup(response.text, "html.parser")
        
        title = soup.title.get_text(strip=True) if soup.title else ""
        
        # Attempt to extract publication date from meta tags (heuristic)
        pub_date = ""
        for meta in soup.find_all("meta"):
            if meta.get("property", "").lower() in ["article:published_time", "og:published_time"]:
                pub_date = meta.get("content", "")
                break
        
        paragraphs = soup.find_all("p")
        content = " ".join([p.get_text(strip=True) for p in paragraphs])
        
        full_text = f"Title: {title}\nPublication Date: {pub_date}\nContent: {content}"
        return clean_text(full_text)
    except Exception as e:
        print(f"Error crawling {url}: {e}")
        return None

def crawl_site(site, max_pages=10):
    texts = []
    visited = set()
    
    # Crawl the homepage first
    homepage_text = crawl_page(site)
    if homepage_text:
        texts.append(f"---\nSource: {site}\n{homepage_text}\n")
    visited.add(site)
    
    try:
        response = requests.get(site, timeout=10)
        if response.status_code != 200:
            return texts
        soup = BeautifulSoup(response.text, "html.parser")
        
        links = set()
        for a in soup.find_all('a', href=True):
            link = a['href']
            if not link.startswith('http'):
                link = urljoin(site, link)
            # Ensure the link is internal by checking the site's domain
            domain = site.split("//")[-1].split("/")[0]
            if domain in link:
                links.add(link)
        
        count = 0
        for link in links:
            if count >= max_pages:
                break
            if link in visited:
                continue
            visited.add(link)
            page_text = crawl_page(link)
            if page_text:
                texts.append(f"---\nSource: {link}\n{page_text}\n")
                count += 1
    except Exception as e:
        print(f"Error crawling site {site}: {e}")
    
    return texts

# Main crawler: iterate over categories and websites, then save into a text file per category
for category, websites in categories.items():
    collected_texts = []
    print(f"Crawling category: {category}")
    for site in websites:
        print(f"  Processing site: {site}")
        site_texts = crawl_site(site, max_pages=10)
        collected_texts.extend(site_texts)
    
    # collected text to a file for this category
    file_path = os.path.join(output_dir, f"{category}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(collected_texts))
    print(f"  Saved data to {file_path}")

print("Crawling and preprocessing completed.")

# Zipping the output directory for direct download in Kaggle
zip_filename = "TextWorldCorpus"
shutil.make_archive(zip_filename, 'zip', output_dir)
print(f"Zipped folder created: {zip_filename}.zip")
