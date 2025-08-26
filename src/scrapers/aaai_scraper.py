import requests
from bs4 import BeautifulSoup
import csv
import os


# Scrape issues from 1 to 1000
import re
BASE_URL = "https://ojs.aaai.org"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
}

# Create absolute path to ensure the file can be created
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
data_raw_dir = os.path.join(project_root, "data", "raw")

# Create directory if it doesn't exist
os.makedirs(data_raw_dir, exist_ok=True)

output_file = os.path.join(data_raw_dir, "aaai25_papers_authors_split.csv")
print(f"Output file will be saved to: {output_file}")

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["title", "article_link", "author", "pdf_link", "year", "issue_number"])
    writer.writeheader()
    total_rows = 0
    for issue_num in range(1, 1001):
        url = f"https://ojs.aaai.org/index.php/AAAI/issue/view/{issue_num}"
        try:
            # Use a tuple for timeout: (connect timeout, read timeout)
            response = requests.get(url, headers=headers, timeout=(5, 30))
            if response.status_code != 200:
                print(f"Issue {issue_num}: Not found (status {response.status_code})")
                continue
            # Avoid printing raw HTML (very large). Parse directly and log concise info.
            soup = BeautifulSoup(response.text, "html.parser")
            # Extract year from <div class="published"><span class="value">YYYY-MM-DD</span></div>
            year = 2025
            soup = BeautifulSoup(response.text, "html.parser")
            published_div = soup.find("div", {"class": "published"})
            if published_div:
                value_span = published_div.select_one("span.value")
                if value_span:
                    date_text = value_span.get_text(strip=True)
                    date_match = re.match(r"(\d{4})-(\d{2})-(\d{2})", date_text)
                    if date_match:
                        year = int(date_match.group(1))
            for article_div in soup.find_all("div", class_="obj_article_summary"):
                h3 = article_div.find("h3", class_="title")
                a = h3.find("a") if h3 else None
                title = a.get_text(strip=True) if a else None
                article_link = a['href'] if a and a.has_attr('href') else None
                if article_link and not article_link.startswith("http"):
                    article_link = BASE_URL + article_link
                authors_div = article_div.find("div", class_="authors")
                authors = authors_div.get_text(strip=True) if authors_div else None
                pdf_link = None
                galleys = article_div.find("ul", class_="galleys_links")
                if galleys:
                    pdf_a = galleys.find("a", class_="obj_galley_link pdf")
                    if pdf_a and pdf_a.has_attr('href'):
                        pdf_link = pdf_a['href']
                        if not pdf_link.startswith("http"):
                            pdf_link = BASE_URL + pdf_link
                if title and article_link and authors and pdf_link:
                    authors_list = [a.strip() for a in authors.split(",")]
                    for author in authors_list:
                        writer.writerow({
                            "title": title,
                            "article_link": article_link,
                            "author": author,
                            "pdf_link": pdf_link,
                            "year": year,
                            "issue_number": issue_num
                        })
                        total_rows += 1
            print(f"✅ Finished issue {issue_num}, total rows so far: {total_rows}")
        except Exception as e:
            print(f"Error processing issue {issue_num}: {e}")

print(f"✅ Extracted {total_rows} author-paper rows across all issues and saved to {output_file}")
