import os
import time
import random
import json
import re
import pandas as pd
import tldextract
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import networkx as nx
import numpy as np
from fuzzywuzzy import process, fuzz
from collections import defaultdict, deque
from datetime import datetime
import difflib # Added for fuzzy_wikipedia_search

# Streamlit specific imports
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import torch
import mwparserfromhell # Added as it was used in clean_wiki_markup
import requests # Added for web requests

# --- Global Constants and Mappings ---
# CONFIG
SAVE_EVERY = 5 # How often to save progress to session state (and implicitly, to display)
MAX_CRAWL_DEPTH_DEFAULT = 3 # A more reasonable default for a demo app
SEED_USER_ID_DEFAULT = "0b_Q5gcAAAAJ" # Example: Geoffrey Hinton
MAX_CRAWL_SECONDS_DEFAULT = 300 # 5 minutes default for a demo app

# Mappings (as previously defined)
domain_to_institution = {
    "nus.edu.sg": "National University of Singapore", "a-star.edu.sg": "A*STAR (Agency for Science, Technology and Research)",
    "mit.edu": "Massachusetts Institute of Technology", "cam.ac.uk": "University of Cambridge",
    "imperial.ac.uk": "Imperial College London", "ucl.ac.uk": "University College London",
    "ucla.edu": "University of California, Los Angeles", "stanford.edu": "Stanford University",
    "harvard.edu": "Harvard University", "berkeley.edu": "University of California, Berkeley",
    "utoronto.ca": "University of Toronto", "queensu.ca": "Queen's University",
    "unimelb.edu.au": "University of Melbourne", "sydney.edu.au": "University of Sydney",
    "monash.edu": "Monash University", "kaust.edu.sa": "King Abdullah University of Science and Technology",
    "tsinghua.edu.cn": "Tsinghua University", "pku.edu.cn": "Peking University",
    "ethz.ch": "ETH Zurich", "epfl.ch": "École Polytechnique Fédérale de Lausanne",
    "google.com": "Google LLC", "microsoft.com": "Microsoft Corporation", "ibm.com": "IBM Corporation",
    "amazon.com": "Amazon.com, Inc.", "facebook.com": "Meta Platforms, Inc.", "openai.com": "OpenAI",
    "flatironinstitute.org": "Flatiron Institute", "ed.ac.uk": "University of Edinburgh",
    "bham.ac.uk": "University of Birmingham", "ncl.ac.uk": "Newcastle University",
    "manchester.ac.uk": "University of Manchester", "lboro.ac.uk": "Loughborough University",
    "deshawresearch.com": "D. E. Shaw Research",
}

suffix_country_map = {
    'edu.sg': 'sg', 'gov.sg': 'sg', 'ac.uk': 'uk', 'edu.au': 'au', 'edu.cn': 'cn', 'edu.in': 'in',
    'edu.ca': 'ca', 'edu': 'us', 'gov': 'us', 'ac.jp': 'jp', 'ac.kr': 'kr', 'ac.at': 'at',
    'ac.be': 'be', 'be': 'be', 'ac.nz': 'nz', 'com': 'unknown', 'org': 'unknown', 'net': 'unknown',
    'sg': 'sg', 'uk': 'uk', 'us': 'us', 'fr': 'fr', 'de': 'de', 'at': 'at', 'ca': 'ca',
    'au': 'au', 'cn': 'cn', 'jp': 'jp', 'kr': 'kr',
}

synonym_map = {"ml": "machine learning", "ai": "artificial intelligence"}

multi_word_phrases = [
    "machine learning", "artificial intelligence", "quantum chemistry", "computational materials science",
    "deep learning", "molecular dynamics", "homogeneous catalysis", "organometallic chemistry",
    "polymer chemistry", "drug discovery", "genome engineering", "synthetic biology",
    "protein engineering", "metabolic engineering", "quantum computing", "density functional theory"
]

# For Wikipedia classification
candidate_labels = ["researcher", "scientist", "engineer", "professor", "academic", "artist", "musician", "politician", "athlete", "writer"]
researcher_labels = ["researcher", "scientist", "engineer", "professor", "academic"]

# --- Streamlit Session State Initialization ---
# Initialize session state variables if they don't exist
if 'all_profiles' not in st.session_state:
    st.session_state.all_profiles = []
if 'visited_depths' not in st.session_state:
    st.session_state.visited_depths = {}
if 'graph' not in st.session_state:
    st.session_state.graph = nx.Graph()
if 'queue' not in st.session_state:
    st.session_state.queue = deque()
if '_iclr_df' not in st.session_state:
    st.session_state._iclr_df = None
if '_neurips_df' not in st.session_state:
    st.session_state._neurips_df = None
if 'status_message' not in st.session_state:
    st.session_state.status_message = "Ready to start crawling."
if 'is_crawling' not in st.session_state:
    st.session_state.is_crawling = False
if 'progress_value' not in st.session_state:
    st.session_state.progress_value = 0
if 'progress_text' not in st.session_state:
    st.session_state.progress_text = "0 profiles scraped."
if 'iclr_file_uploaded' not in st.session_state:
    st.session_state.iclr_file_uploaded = None
if 'neurips_file_uploaded' not in st.session_state:
    st.session_state.neurips_file_uploaded = None

# --- Cached Resources (Models) ---
@st.cache_resource
def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

@st.cache_resource
def load_classifier(device_name):
    with st.spinner(f"Loading zero-shot classification model on {device_name}..."):
        return pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=0 if device_name in ["cuda", "mps"] else -1
        )

@st.cache_resource
def load_sentence_transformer_model():
    with st.spinner("Loading SentenceTransformer model..."):
        return SentenceTransformer("all-MiniLM-L6-v2")

# Load models
device = get_device()
classifier = load_classifier(device)
model = load_sentence_transformer_model() # Not used in current crawl logic, but kept for future clustering

# --- Helper Functions ---

def clean_wiki_markup(raw_text):
    wikicode = mwparserfromhell.parse(raw_text)
    for template in wikicode.filter_templates():
        if template.name.lower().strip() in ['ubl', 'plainlist', 'flatlist', 'hlist']:
            items = [mwparserfromhell.parse(str(param.value)).strip_code().strip() for param in template.params]
            items = [re.sub(r'\\[\\[]|\\[\\]]', '', item) for item in items if item]
            wikicode.replace(template, "; ".join(items))
    for link in wikicode.filter_wikilinks():
        link_text = link.text if link.text else link.title
        wikicode.replace(link, str(link_text))
    for template in wikicode.filter_templates():
        wikicode.remove(template)
    cleaned_text = wikicode.strip_code()
    cleaned_text = re.sub(r';+\s*', '; ', cleaned_text.replace('\n', '; ')).strip()
    cleaned_text = re.sub(r'[{}\|]', '', cleaned_text).strip('; ').strip()
    return cleaned_text

def fuzzy_wikipedia_search(name, threshold=0.90, max_results=5):
    S = requests.Session()
    S.headers.update({'User-Agent': 'Mozilla/5.0 (compatible; WikiInfoBot/1.0)'})
    URL = "https://en.wikipedia.org/w/api.php"
    PARAMS = {"action": "query", "list": "search", "srsearch": name, "srlimit": max_results, "format": "json"}
    retries = 3
    for _ in range(retries):
        try:
            response = S.get(url=URL, params=PARAMS)
            response.raise_for_status()
            data = response.json()
            break
        except (requests.exceptions.RequestException, ValueError, Exception) as e:
            st.warning(f"Wikipedia search failed for '{name}', retrying... Error: {e}")
            time.sleep(1)
    else:
        return None
    search_results = data.get("query", {}).get("search", [])
    best_match, best_score = None, 0
    for result in search_results:
        title = result['title']
        score = difflib.SequenceMatcher(None, name.lower(), title.lower()).ratio()
        if score > best_score:
            best_score, best_match = score, title
    return best_match if best_score >= threshold else None

def get_wikipedia_summary(page_title):
    S = requests.Session()
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{page_title}"
    S.headers.update({'User-Agent': 'Mozilla/5.0 (compatible; WikiInfoBot/1.0)'})
    try:
        response = S.get(url)
        response.raise_for_status()
        data = response.json()
        return data.get("extract", "")
    except (requests.exceptions.RequestException, ValueError) as e:
        st.warning(f"Wikipedia summary failed for '{page_title}': {e}")
        return ""

def get_selected_infobox_fields(page_title, fields_to_extract):
    S = requests.Session()
    S.headers.update({'User-Agent': 'Mozilla/5.0 (compatible; WikiInfoBot/1.0)'})
    URL = "https://en.wikipedia.org/w/api.php"
    PARAMS = {"action": "query", "format": "json", "titles": page_title,
              "prop": "revisions", "rvprop": "content", "rvslots": "main"}
    retries = 3
    for _ in range(retries):
        try:
            response = S.get(url=URL, params=PARAMS)
            response.raise_for_status()
            data = response.json()
            break
        except (requests.exceptions.RequestException, ValueError, Exception) as e:
            st.warning(f"Wikipedia infobox failed for '{page_title}', retrying... Error: {e}")
            time.sleep(1)
    else:
        return None, None
    page = next(iter(data['query']['pages'].values()))
    if "missing" in page or 'revisions' not in page:
        return None, None
    matched_title = page.get('title', page_title)
    wikitext = page['revisions'][0]['slots']['main']['*']
    wikicode = mwparserfromhell.parse(wikitext)
    infobox = next((t for t in wikicode.filter_templates() if t.name.lower().strip().startswith("infobox")), None)
    extracted = {}
    if infobox:
        for key in fields_to_extract:
            try:
                val = infobox.get(key).value.strip()
                extracted[key] = clean_wiki_markup(str(val))
            except Exception:
                extracted[key] = ""
    else:
        extracted = {k: "" for k in fields_to_extract}
    return extracted, matched_title

def classify_summary(summary):
    summary = summary.strip()
    if not summary:
        return False
    classification = classifier(summary, candidate_labels)
    top_label = classification['labels'][0].lower()
    return top_label in [label.lower() for label in researcher_labels]

def get_author_wikipedia_info(author_name):
    fields = ["birth_name", "name", "birth_date", "birth_place", "death_date", "death_place",
              "fields", "work_institution", "alma_mater", "notable_students", "thesis_title",
              "thesis_year", "thesis_url", "known_for", "awards"]

    matched_title = fuzzy_wikipedia_search(author_name)
    if matched_title is None:
        info = {k: "" for k in fields}
        info["deceased"] = False
        info["wiki_summary"] = ""
        info["is_researcher_ml"] = False
        info["matched_title"] = None
        return info

    info, matched_title = get_selected_infobox_fields(matched_title, fields)
    if info is None:
        info = {k: "" for k in fields}
        info["deceased"] = False
    else:
        info["deceased"] = bool(info.get("death_date"))
    summary = get_wikipedia_summary(matched_title)
    info["wiki_summary"] = summary
    info["is_researcher_ml"] = classify_summary(summary)
    info["matched_title"] = matched_title

    return info

def normalize_interest_phrases(raw_text):
    interests = [s.strip().lower() for s in raw_text.split(",") if s.strip()]
    processed = []
    for interest in interests:
        interest = synonym_map.get(interest, interest)
        for phrase in multi_word_phrases:
            if interest == phrase:
                interest = phrase.replace(" ", "_")
                break
        processed.append(interest)
    return processed

@st.cache_resource(ttl=3600) # Cache for 1 hour
def get_driver():
    st.info("Initializing Selenium WebDriver (this may take a moment)...")
    
    # Define paths for Chrome and Chromedriver on Streamlit Cloud
    CHROME_PATH = "/usr/bin/google-chrome" # Path where Google Chrome will be installed
    CHROMEDRIVER_PATH = "/usr/local/bin/chromedriver" # Path where Chromedriver will be installed

    # Check if Chrome and Chromedriver are already downloaded and executable
    if not os.path.exists(CHROME_PATH) or not os.path.exists(CHROMEDRIVER_PATH):
        st.warning("Chromium or Chromedriver not found. Attempting to download...")
        # Download Google Chrome (Debian package)
        try:
            # Get latest stable Chrome version
            chrome_version_output = os.popen(
                "wget -q -O - https://googlechromelabs.github.io/chrome-for-testing/last-known-good-versions-with-downloads.json | grep -oP '\"LATEST_RELEASE_([0-9.]+)\"' | head -1 | cut -d'_' -f3 | cut -d'\"' -f1"
            ).read().strip()

            if not chrome_version_output:
                st.error("Could not determine latest Chrome stable version.")
                raise Exception("Chrome version determination failed.")

            # Download URL for the latest stable Chrome (x86_64, .deb)
            # This URL is for Google Chrome itself, which is often easier to install than chromium-browser
            chrome_download_url = f"https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb"
            
            st.info(f"Downloading Google Chrome stable from {chrome_download_url}...")
            os.system(f"wget {chrome_download_url} -O /tmp/google-chrome-stable_current_amd64.deb")
            os.system("sudo dpkg -i /tmp/google-chrome-stable_current_amd64.deb || sudo apt-get install -fy")
            
            # This will install Chrome to /opt/google/chrome/google-chrome,
            # we want a symlink to /usr/bin for consistent access
            if not os.path.exists("/usr/bin/google-chrome"):
                os.system("sudo ln -s /opt/google/chrome/google-chrome /usr/bin/google-chrome")
            
            if not os.path.exists(CHROME_PATH):
                st.error(f"Google Chrome not found at {CHROME_PATH} after installation.")
                raise Exception("Google Chrome installation failed.")
            st.success("Google Chrome installed!")

            # Download Chromedriver matching the installed Chrome version
            # Fetch the exact version of the installed Chrome
            installed_chrome_version_output = os.popen("google-chrome --version").read().strip()
            # Extract version number like "126.0.6478.182"
            installed_chrome_version = re.search(r"(\d+\.\d+\.\d+\.\d+)", installed_chrome_version_output)
            if not installed_chrome_version:
                st.error("Could not determine installed Chrome version to match Chromedriver.")
                raise Exception("Installed Chrome version determination failed.")
            installed_chrome_version = installed_chrome_version.group(1)
            
            # Use Chrome for Testing (CfT) API to find compatible Chromedriver
            chromedriver_api_url = f"https://googlechromelabs.github.io/chrome-for-testing/last-known-good-versions-with-downloads.json"
            response = requests.get(chromedriver_api_url)
            response.raise_for_status()
            versions_data = response.json()
            
            chromedriver_url = None
            for version_entry in versions_data['versions']:
                if version_entry['version'] == installed_chrome_version:
                    for download in version_entry['downloads']['chromedriver']:
                        if download['platform'] == 'linux64':
                            chromedriver_url = download['url']
                            break
                if chromedriver_url:
                    break
            
            if not chromedriver_url:
                st.error(f"Could not find a matching Chromedriver for Chrome version {installed_chrome_version}")
                raise Exception("Matching Chromedriver URL not found.")

            st.info(f"Downloading Chromedriver from {chromedriver_url}...")
            os.system(f"wget {chromedriver_url} -O /tmp/chromedriver.zip")
            os.system(f"unzip -o /tmp/chromedriver.zip -d /tmp/chromedriver_extracted")
            os.system(f"sudo mv /tmp/chromedriver_extracted/chromedriver {CHROMEDRIVER_PATH}")
            os.system(f"sudo chmod +x {CHROMEDRIVER_PATH}")

            if not os.path.exists(CHROMEDRIVER_PATH):
                st.error(f"Chromedriver not found at {CHROMEDRIVER_PATH} after installation.")
                raise Exception("Chromedriver installation failed.")
            st.success("Chromedriver installed!")

        except Exception as e:
            st.error(f"Failed to set up Chrome and Chromedriver dynamically: {e}")
            st.stop() # Stop the app if setup fails

    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-extensions")
    options.add_argument("--proxy-server='direct://'")
    options.add_argument("--proxy-bypass-list=*")
    options.add_argument("--start-maximized")
    options.add_argument("--no-zygote")

    try:
        driver = webdriver.Chrome(
            executable_path=CHROMEDRIVER_PATH,
            options=options,
            # service_args=["--verbose", "--log-path=/tmp/chromedriver.log"] # Uncomment for debugging
        )
        st.success("Selenium WebDriver initialized successfully!")
    except Exception as e:
        st.error(f"Failed to initialize Chrome driver. Error: {e}")
        st.error("Ensure Chrome and Chromedriver are correctly installed and compatible.")
        st.exception(e) # Display full exception details in the UI
        st.stop() # Stop the app if the driver fails to initialize
    return driver


def extract_profile(driver, user_id, depth):
    url = f"https://scholar.google.com/citations?hl=en&user={user_id}"
    driver.get(url)
    time.sleep(random.uniform(1.5, 3.0))
    soup = BeautifulSoup(driver.page_source, "html.parser")

    name = soup.select_one("#gsc_prf_in").get_text(strip=True) if soup.select_one("#gsc_prf_in") else "Unknown"
    position = soup.select_one(".gsc_prf_il").get_text(strip=True) if soup.select_one(".gsc_prf_il") else "Unknown"
    email = soup.select_one("#gsc_prf_ivh").get_text(strip=True) if soup.select_one("#gsc_prf_ivh") else "Unknown"
    interests_raw = ", ".join(tag.text for tag in soup.select("#gsc_prf_int a")) if soup.select("#gsc_prf_int a") else ""
    interest_phrases = normalize_interest_phrases(interests_raw)
    country = infer_country_from_email_field(email)
    institution = get_institution_from_email(email)

    citations_all = "0"
    h_index_all = "0"
    metrics_rows = soup.select("#gsc_rsb_st tbody tr")
    if metrics_rows and len(metrics_rows) >= 2:
        try:
            citations_all = metrics_rows[0].select("td")[1].text
            h_index_all = metrics_rows[1].select("td")[1].text
        except IndexError:
            pass

    coauthors = []
    for a_tag in soup.select(".gsc_rsb_aa .gsc_rsb_a_desc a"):
        href = a_tag.get("href", "")
        if "user=" in href:
            co_id = href.split("user=")[1].split("&")[0]
            coauthors.append(co_id)
    st.session_state.status_message = f"✅ Found {len(coauthors)} co-authors for {user_id}"
    st.sidebar.caption(st.session_state.status_message) # Display in sidebar

    profile = {
        "user_id": user_id, "name": name, "position": position, "email": email, "country": country,
        "institution": institution, "research_interests": interests_raw, "interest_phrases": interest_phrases,
        "citations_all": citations_all, "h_index_all": h_index_all, "topic_clusters": [],
        "search_depth": depth, "coauthors": coauthors,
        "Participated_in_ICLR": False, "ICLR_Institution": "", # Default values
        "Participated_in_NeurIPS": False, "NeurIPS_Institution": "", # Default values
        "Fuzzy_Matched": False # Default value
    }
    return profile

def get_coauthors_from_profile(driver, user_id):
    """
    Extracts co-author IDs from a given Google Scholar profile page.
    Used for re-queuing if the main queue becomes empty.
    """
    url = f"https://scholar.google.com/citations?hl=en&user={user_id}"
    driver.get(url)
    time.sleep(random.uniform(1.0, 2.0)) # Shorter sleep for just co-authors
    soup = BeautifulSoup(driver.page_source, "html.parser")
    coauthors = []
    for a_tag in soup.select(".gsc_rsb_aa .gsc_rsb_a_desc a"):
        href = a_tag.get("href", "")
        if "user=" in href:
            co_id = href.split("user=")[1].split("&")[0]
            coauthors.append(co_id)
    return coauthors


def infer_country_from_email_field(email_field):
    match = re.search(r"Verified email at ([^\s]+?)(?:\s*-\s*Homepage)?$", email_field)
    if not match:
        return "unknown"
    domain = match.group(1).lower().strip()
    ext = tldextract.extract(domain)
    suffix = ext.suffix.lower()
    return suffix_country_map.get(suffix, suffix_country_map.get(suffix.split('.')[-1], "unknown"))

def get_institution_from_email(email_field):
    match = re.search(r"Verified email at ([^\s]+?)(?:\s*-\s*Homepage)?$", email_field)
    if not match:
        return "Unknown"
    domain = match.group(1).lower().strip()
    for known_domain, institution_name in domain_to_institution.items():
        if domain.endswith(known_domain):
            return institution_name
    return "Unknown"

def fuzzy_match_conference_participation(profile, conf_name, df, name_col='Author', inst_col='Institution', threshold=85):
    authors = df[name_col].dropna().unique()
    authors_lower = [a.lower() for a in authors]

    profile_name = profile.get("name", "").lower()
    if not profile_name:
        profile[f"Participated_in_{conf_name}"] = False
        profile[f"{conf_name}_Institution"] = ""
        return

    match, score = process.extractOne(profile_name, authors_lower, scorer=fuzz.token_sort_ratio)
    if score >= threshold:
        match_idx = authors_lower.index(match)
        matched_author = authors[match_idx]
        matched_institution = df[df[name_col] == matched_author][inst_col].dropna().iloc[0] if inst_col in df.columns else ""
        profile[f"Participated_in_{conf_name}"] = True
        profile[f"{conf_name}_Institution"] = matched_institution
        # st.sidebar.caption(f"✅ [{conf_name}] Match: {profile['name']} → {matched_author} @ {matched_institution}") # Too verbose for sidebar
    else:
        profile[f"Participated_in_{conf_name}"] = False
        profile[f"{conf_name}_Institution"] = ""
        # st.sidebar.caption(f"❌ [{conf_name}] No match for {profile['name']}") # Too verbose for sidebar

def run_fuzzy_matching():
    if st.session_state._iclr_df is None and st.session_state.iclr_file_uploaded is not None:
        try:
            st.session_state._iclr_df = pd.read_parquet(st.session_state.iclr_file_uploaded)
            st.sidebar.info("ICLR Parquet data loaded for matching.")
        except Exception as e:
            st.sidebar.error(f"Error loading ICLR Parquet: {e}")
            st.session_state._iclr_df = None

    if st.session_state._neurips_df is None and st.session_state.neurips_file_uploaded is not None:
        try:
            st.session_state._neurips_df = pd.read_parquet(st.session_state.neurips_file_uploaded)
            st.sidebar.info("NeurIPS Parquet data loaded for matching.")
        except Exception as e:
            st.sidebar.error(f"Error loading NeurIPS Parquet: {e}")
            st.session_state._neurips_df = None

    profiles_to_match = [p for p in st.session_state.all_profiles if not p.get("Fuzzy_Matched")]
    if not profiles_to_match:
        st.sidebar.info("No new profiles to fuzzy match.")
        return

    st.sidebar.info(f"🔄 Running fuzzy matching on {len(profiles_to_match)} new profiles...")
    for profile in profiles_to_match:
        if st.session_state._iclr_df is not None:
            fuzzy_match_conference_participation(profile, "ICLR", st.session_state._iclr_df)
        if st.session_state._neurips_df is not None:
            fuzzy_match_conference_participation(profile, "NeurIPS", st.session_state._neurips_df)
        profile["Fuzzy_Matched"] = True
    st.session_state.status_message = f"✅ Fuzzy matching completed for {len(profiles_to_match)} profiles."
    st.sidebar.success(st.session_state.status_message)


def save_progress_to_session():
    # This function is now mostly a placeholder as data is directly in session_state
    st.session_state.status_message = f"💾 Progress updated in session: {len(st.session_state.all_profiles)} profiles"
    st.sidebar.caption(st.session_state.status_message)

def enqueue_user(user_id, depth, parent_id=None):
    if user_id in st.session_state.visited_depths:
        return
    # Check if already in queue to prevent duplicates, (user_id, depth, parent_id) tuple
    if any(user_id == item[0] for item in st.session_state.queue):
        return
    st.session_state.queue.append((user_id, depth, parent_id))

def crawl_bfs_resume(driver, max_crawl_depth, max_crawl_seconds):
    st.session_state.is_crawling = True
    start_time = time.time()
    total_scraped_this_run = 0

    progress_bar_placeholder = st.empty()
    progress_text_placeholder = st.empty()

    visited_ids = set(st.session_state.visited_depths.keys())

    while st.session_state.queue:
        if not st.session_state.is_crawling: # Allow stopping the crawl
            st.info("Crawl paused by user.")
            break

        user_id, depth, parent_id = st.session_state.queue.popleft()
        depth = int(depth)

        elapsed = time.time() - start_time
        if elapsed > max_crawl_seconds:
            st.session_state.status_message = f"🛑 Max crawl time {max_crawl_seconds}s reached, ending crawl."
            st.warning(st.session_state.status_message)
            break

        if user_id in visited_ids:
            st.sidebar.caption(f"Skipping already visited user {user_id}")
            continue

        st.session_state.status_message = f"🔎 Crawling {user_id} at depth {depth}"
        progress_text_placeholder.text(st.session_state.status_message)
        # Calculate progress: (scraped_count) / (scraped_count + queue_size)
        current_scraped_count = len(st.session_state.all_profiles)
        current_queue_size = len(st.session_state.queue)
        if current_scraped_count + current_queue_size > 0:
            st.session_state.progress_value = current_scraped_count / (current_scraped_count + current_queue_size + 1) # +1 to avoid division by zero early
        else:
            st.session_state.progress_value = 0.0
        progress_bar_placeholder.progress(st.session_state.progress_value)
        # st.experimental_rerun() # Removed: Calling this in a loop can cause infinite reruns. Updates will happen on next Streamlit run cycle.

        try:
            profile = extract_profile(driver, user_id, depth)
            st.session_state.visited_depths[user_id] = depth
            visited_ids.add(user_id) # Add to the set of visited IDs

            wiki_info = get_author_wikipedia_info(profile.get("name", ""))
            profile.update({
                "wiki_birth_name": wiki_info.get("birth_name", ""), "wiki_name": wiki_info.get("name", ""),
                "wiki_birth_date": wiki_info.get("birth_date", ""), "wiki_birth_place": wiki_info.get("birth_place", ""),
                "wiki_death_date": wiki_info.get("death_date", ""), "wiki_death_place": wiki_info.get("death_place", ""),
                "wiki_fields": wiki_info.get("fields", ""), "wiki_work_institution": wiki_info.get("work_institution", ""),
                "wiki_alma_mater": wiki_info.get("alma_mater", ""), "wiki_notable_students": wiki_info.get("notable_students", ""),
                "wiki_thesis_title": wiki_info.get("thesis_title", ""), "wiki_thesis_year": wiki_info.get("thesis_year", ""),
                "wiki_thesis_url": wiki_info.get("thesis_url", ""), "wiki_known_for": wiki_info.get("known_for", ""),
                "wiki_awards": wiki_info.get("awards", ""), "wiki_deceased": wiki_info.get("deceased", False),
                "wiki_wiki_summary": wiki_info.get("wiki_summary", ""), "wiki_is_researcher_ml": wiki_info.get("is_researcher_ml", False),
                "wiki_matched_title": wiki_info.get("matched_title", None)
            })

            st.session_state.all_profiles.append(profile)
            total_scraped_this_run += 1

            st.sidebar.success(f"✅ Scraped {profile['name']} | h-index: {profile['h_index_all']} | {profile['country']} | depth {depth}")

            # Add co-authors to queue for next depth
            if depth + 1 <= max_crawl_depth:
                for co_id in profile.get("coauthors", [])[:20]: # Limit co-authors to prevent overwhelming queue
                    enqueue_user(co_id, depth + 1, user_id)

            if total_scraped_this_run % SAVE_EVERY == 0:
                save_progress_to_session()
                # Run fuzzy matching periodically
                if len(st.session_state.all_profiles) > 0 and (total_scraped_this_run % 20 == 0): # Run every 20 profiles
                    run_fuzzy_matching()

        except Exception as e:
            st.error(f"❌ Error scraping {user_id}: {e}")
            st.session_state.status_message = f"Error scraping {user_id}: {e}"
            # st.experimental_rerun() # Removed: Calling this in a loop can cause infinite reruns.

    st.session_state.status_message = f"✅ BFS crawl finished with {total_scraped_this_run} new profiles."
    st.success(st.session_state.status_message)
    run_fuzzy_matching() # Final fuzzy matching run
    st.session_state.is_crawling = False
    st.session_state.progress_value = 1.0
    progress_bar_placeholder.progress(st.session_state.progress_value)
    progress_text_placeholder.text("Crawl completed!")
    # st.experimental_rerun() # Removed: Let Streamlit handle the natural rerun

# --- Streamlit UI Layout ---
st.set_page_config(layout="wide", page_title="Scholar Profile Crawler")

st.title("👨‍🎓 Google Scholar Profile Crawler")
st.markdown("Crawl Google Scholar profiles, extract information, and find conference participations.")

# Sidebar for controls and status
st.sidebar.header("Controls")
seed_user_id = st.sidebar.text_input("Seed User ID", value=SEED_USER_ID_DEFAULT)
max_crawl_depth = st.sidebar.number_input("Max Crawl Depth", min_value=1, value=MAX_CRAWL_DEPTH_DEFAULT)
max_crawl_seconds = st.sidebar.number_input("Max Crawl Time (seconds)", min_value=60, value=MAX_CRAWL_SECONDS_DEFAULT)

st.sidebar.subheader("Conference Data Upload (Optional)")
iclr_file = st.sidebar.file_uploader("Upload ICLR Parquet (iclr_2020_2025_combined_data.parquet)", type=["parquet"], key="iclr_uploader")
neurips_file = st.sidebar.file_uploader("Upload NeurIPS Parquet (neurips_2020_2024_combined_data.parquet)", type=["parquet"], key="neurips_uploader")

# Load uploaded files into session state only once per upload
if iclr_file is not st.session_state.iclr_file_uploaded:
    st.session_state.iclr_file_uploaded = iclr_file
    if iclr_file is not None:
        with st.spinner("Loading ICLR data..."):
            try:
                st.session_state._iclr_df = pd.read_parquet(iclr_file)
                st.sidebar.success("ICLR data loaded!")
            except Exception as e:
                st.sidebar.error(f"Error loading ICLR Parquet: {e}")
                st.session_state._iclr_df = None
    else:
        st.session_state._iclr_df = None # File was unselected

if neurips_file is not st.session_state.neurips_file_uploaded:
    st.session_state.neurips_file_uploaded = neurips_file
    if neurips_file is not None:
        with st.spinner("Loading NeurIPS data..."):
            try:
                st.session_state._neurips_df = pd.read_parquet(neurips_file)
                st.sidebar.success("NeurIPS data loaded!")
            except Exception as e:
                st.sidebar.error(f"Error loading NeurIPS Parquet: {e}")
                st.session_state._neurips_df = None
    else:
        st.session_state._neurips_df = None # File was unselected


st.sidebar.subheader("Actions")
col1, col2 = st.sidebar.columns(2)

if col1.button("Start/Resume Crawl", disabled=st.session_state.is_crawling):
    if not st.session_state.queue and not st.session_state.all_profiles:
        # Initial start: Queue the seed user
        enqueue_user(seed_user_id, 0)
        st.session_state.status_message = f"Starting crawl from seed: {seed_user_id}"
    elif not st.session_state.queue and st.session_state.all_profiles:
        # Resume with last scraped users' co-authors if queue is empty
        # This part requires fetching co-authors *outside* the main crawl loop
        st.sidebar.info("Queue empty. Attempting to re-populate queue from recent profiles.")
        recent_profiles = st.session_state.all_profiles[-20:] # Look at last 20 profiles
        driver_instance = get_driver() # Get driver for this operation

        coauthors_to_add = set()
        for p in recent_profiles:
            if p["search_depth"] < max_crawl_depth: # Only add co-authors if not at max depth
                try:
                    new_coauthors = get_coauthors_from_profile(driver_instance, p["user_id"])
                    for co_id in new_coauthors:
                        if co_id not in st.session_state.visited_depths and co_id not in {q[0] for q in st.session_state.queue}:
                            coauthors_to_add.add((co_id, p["search_depth"] + 1, p["user_id"]))
                except Exception as e:
                    st.sidebar.warning(f"Failed to get co-authors for {p['user_id']} to resume: {e}")
        
        if coauthors_to_add:
            for co_id, depth, parent_id in coauthors_to_add:
                enqueue_user(co_id, depth, parent_id)
            st.sidebar.success(f"Queue repopulated with {len(coauthors_to_add)} co-authors.")
        else:
            st.sidebar.warning(f"Could not repopulate queue. Re-queuing seed user {seed_user_id}.")
            enqueue_user(seed_user_id, 0)


    st.session_state.is_crawling = True
    st.session_state.status_message = "Crawl started..."
    # st.experimental_rerun() # Removed by previous instructions
    st.rerun() # Use st.rerun() if you really need to force a rerun and your Streamlit version supports it

if col2.button("Stop Crawl", disabled=not st.session_state.is_crawling):
    st.session_state.is_crawling = False
    st.session_state.status_message = "Crawl stopping soon..."
    # st.experimental_rerun() # Removed by previous instructions
    st.rerun() # Use st.rerun() if you really need to force a rerun and your Streamlit version supports it

if st.sidebar.button("Reset All Data"):
    st.session_state.all_profiles = []
    st.session_state.visited_depths = {}
    st.session_state.graph = nx.Graph()
    st.session_state.queue = deque()
    st.session_state._iclr_df = None
    st.session_state._neurips_df = None
    st.session_state.iclr_file_uploaded = None # Reset uploaded file state
    st.session_state.neurips_file_uploaded = None # Reset uploaded file state
    st.session_state.status_message = "All data reset. Ready to start fresh."
    st.session_state.is_crawling = False
    st.session_state.progress_value = 0
    st.session_state.progress_text = "0 profiles scraped."
    st.rerun() # Use st.rerun() if you really need to force a rerun and your Streamlit version supports it

st.sidebar.info(f"Current Status: {st.session_state.status_message}")
st.sidebar.progress(st.session_state.progress_value, text=st.session_state.progress_text)
st.sidebar.write(f"Profiles Scraped: {len(st.session_state.all_profiles)}")
st.sidebar.write(f"Queue Size: {len(st.session_state.queue)}")

# Main content area
st.header("Crawled Profiles")
if st.session_state.all_profiles:
    df_display = pd.DataFrame(st.session_state.all_profiles)
    st.dataframe(df_display, use_container_width=True)
else:
    st.info("No profiles scraped yet. Start the crawl from the sidebar!")

# Run the crawl if the state indicates it should be running
if st.session_state.is_crawling:
    with st.spinner("Crawling in progress..."):
        driver = get_driver() # Get the cached driver instance
        crawl_bfs_resume(driver, max_crawl_depth, max_crawl_seconds)
    # After crawl_bfs_resume returns (either finished or stopped by user/time),
    # ensure UI reflects the final state
    st.rerun() # Use st.rerun() if you really need to force a rerun and your Streamlit version supports it

st.header("Raw Data (for debugging)")
with st.expander("Show Raw Session State"):
    st.json({
        "all_profiles_count": len(st.session_state.all_profiles),
        "visited_depths_count": len(st.session_state.visited_depths),
        "queue_size": len(st.session_state.queue),
        "is_crawling": st.session_state.is_crawling,
        "iclr_df_loaded": st.session_state._iclr_df is not None,
        "neurips_df_loaded": st.session_state._neurips_df is not None,
        "iclr_file_uploaded_status": "present" if st.session_state.iclr_file_uploaded else "not uploaded",
        "neurips_file_uploaded_status": "present" if st.session_state.neurips_file_uploaded else "not uploaded",
        # Displaying full profiles/queue can be very large, use with caution for large datasets
        # "all_profiles_sample": st.session_state.all_profiles[:5],
        # "queue_sample": list(st.session_state.queue)[:5]
    })
