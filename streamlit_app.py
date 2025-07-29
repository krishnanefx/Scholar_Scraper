import streamlit as st
import subprocess
import sys
import os
import time
import random
import json
import re
import pandas as pd
import tldextract
from bs4 import BeautifulSoup
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
from fuzzywuzzy import process, fuzz
from collections import Counter, defaultdict, deque
from datetime import datetime

import requests
import mwparserfromhell
import difflib
import torch
from transformers import pipeline

# Import Playwright specific types for clarity
from playwright.sync_api import sync_playwright, BrowserContext, Page

# --- Set environment variable for tokenizers (important for Hugging Face models) ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Playwright Installation (Temporary - NOT FOR PRODUCTION) ---
# This part ensures Playwright browsers are installed.
# Using session_state to run it only once per deployment/session for efficiency.
if 'playwright_browsers_installed' not in st.session_state:
    st.session_state.playwright_browsers_installed = False

if not st.session_state.playwright_browsers_installed:
    st.info("Attempting to install Playwright browsers. This might take a moment...")
    try:
        playwright_cache_dir = os.path.expanduser("~/.cache/ms-playwright")
        
        # Check if the cache directory exists and contains some files (implies browsers might be there)
        # This check is heuristic. A more robust check might involve 'playwright install --dry-run'
        if not os.path.exists(playwright_cache_dir) or not os.listdir(playwright_cache_dir):
            command = [sys.executable, "-m", "playwright", "install"]
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            
            st.success("Playwright browsers installed successfully!")
            st.code(result.stdout)
            st.session_state.playwright_browsers_installed = True
            
        else:
            st.success("Playwright browsers already appear to be installed.")
            st.session_state.playwright_browsers_installed = True

    except subprocess.CalledProcessError as e:
        st.error(f"Failed to install Playwright browsers. Error: {e.stderr}")
        st.code(e.stdout)
        st.code(e.stderr)
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during Playwright browser installation: {e}")
        st.stop()
# --- End Playwright Installation Block ---

# --- Global Constants and Initializations ---

# Use st.cache_resource for heavy objects like models and browser instances
@st.cache_resource
def get_device_cached():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

device = get_device_cached()

@st.cache_resource
def get_classifier():
    return pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=0 if device in ["cuda", "mps"] else -1
    )

classifier = get_classifier()

researcher_labels = [
    "researcher", "scientist", "computer scientist", "AI researcher", "machine learning researcher",
    "academic", "professor", "engineer", "data scientist", "mathematician", "physicist",
    "chemist", "biologist", "linguist", "university lecturer", "postdoctoral researcher",
    "doctoral student", "technical researcher", "scientific author", "science writer"
]

non_research_labels = [
    "comedian", "actor", "actress", "musician", "singer", "rapper", "sportsman", "athlete",
    "footballer", "basketball player", "tennis player", "politician", "diplomat", "lawyer",
    "businessman", "businesswoman", "entrepreneur", "journalist", "influencer", "youtuber",
    "television presenter", "celebrity", "movie director", "film producer", "screenwriter",
    "artist", "painter", "fashion designer", "model", "novelist", "poet", "author",
    "motivational speaker", "podcaster", "public speaker", "filmmaker"
]

candidate_labels = researcher_labels + non_research_labels

# Define all expected columns for the DataFrame to ensure consistency
# Initializing default values for new columns that might be added during processing
EXPECTED_COLUMNS = [
    "user_id", "name", "position", "email", "country", "institution", "research_interests",
    "interest_phrases", "citations_all", "h_index_all", "topic_clusters", "search_depth",
    "Participated_in_ICLR", "ICLR_Matched_Name", "ICLR_Institution", # Added ICLR_Institution here
    "Participated_in_NeurIPS", "NeurIPS_Matched_Name", "NeurIPS_Institution", # Added NeurIPS_Institution here
    "wiki_birth_name", "wiki_name", "wiki_birth_date", "wiki_birth_place",
    "wiki_death_date", "wiki_death_place", "wiki_fields", "wiki_work_institution",
    "wiki_alma_mater", "wiki_notable_students", "wiki_thesis_title", "wiki_thesis_year",
    "wiki_thesis_url", "wiki_known_for", "wiki_awards", "wiki_deceased",
    "wiki_wiki_summary", "wiki_is_researcher_ml", "wiki_matched_title",
    "coauthors", "Fuzzy_Matched"
]


# === Mappings ===
domain_to_institution = {
    "nus.edu.sg": "National University of Singapore",
    "a-star.edu.sg": "A*STAR (Agency for Science, Technology and Research)",
    "mit.edu": "Massachusetts Institute of Technology",
    "cam.ac.uk": "University of Cambridge",
    "imperial.ac.uk": "Imperial College London",
    "ucl.ac.uk": "University College London",
    "ucla.edu": "University of California, Los Angeles",
    "stanford.edu": "Stanford University",
    "harvard.edu": "Harvard University",
    "berkeley.edu": "University of California, Berkeley",
    "utoronto.ca": "University of Toronto",
    "queensu.ca": "Queen's University",
    "unimelb.edu.au": "University of Melbourne",
    "sydney.edu.au": "University of Sydney",
    "monash.edu": "Monash University",
    "kaust.edu.sa": "King Abdullah University of Science and Technology",
    "tsinghua.edu.cn": "Tsinghua University",
    "pku.edu.cn": "Peking University",
    "ethz.ch": "ETH Zurich",
    "epfl.ch": "École Polytechnique Fédérale de Lausanne",
    "google.com": "Google LLC",
    "microsoft.com": "Microsoft Corporation",
    "ibm.com": "IBM Corporation",
    "amazon.com": "Amazon.com, Inc.",
    "facebook.com": "Meta Platforms, Inc.",
    "openai.com": "OpenAI",
    "flatironinstitute.org": "Flatiron Institute",
    "ed.ac.uk": "University of Edinburgh",
    "bham.ac.uk": "University of Birmingham",
    "ncl.ac.uk": "Newcastle University",
    "manchester.ac.uk": "University of Manchester",
    "lboro.ac.uk": "Loughborough University",
    "deshawresearch.com": "D. E. Shaw Research",
    # Add more as needed...
}

suffix_country_map = {
    'edu.sg': 'sg',
    'gov.sg': 'sg',
    'ac.uk': 'uk',
    'edu.au': 'au',
    'edu.cn': 'cn',
    'edu.in': 'in',
    'edu.ca': 'ca',
    'edu': 'us',
    'gov': 'us',
    'ac.jp': 'jp',
    'ac.kr': 'kr',
    'ac.at': 'at',
    'ac.be': 'be',
    'ac.nz': 'nz',
    'com': 'unknown',
    'org': 'unknown',
    'net': 'unknown',
    'sg': 'sg',
    'uk': 'uk',
    'us': 'us',
    'fr': 'fr',
    'de': 'de',
    'at': 'at',
    'ca': 'ca',
    'au': 'au',
    'cn': 'cn',
    'jp': 'jp',
    'kr': 'kr',
    # Add more as needed...
}

synonym_map = {
    "ml": "machine learning",
    "ai": "artificial intelligence",
}

multi_word_phrases = [
    "machine learning", "artificial intelligence", "quantum chemistry",
    "computational materials science", "deep learning", "molecular dynamics",
    "homogeneous catalysis", "organometallic chemistry", "polymer chemistry",
    "drug discovery", "genome engineering", "synthetic biology", "protein engineering",
    "metabolic engineering", "quantum computing", "density functional theory"
]

# --- Functions ---

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
            st.warning(f"Wikipedia search failed for '{name}' (attempt {_ + 1}/{retries}): {e}")
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
        st.warning(f"Failed to get Wikipedia summary for '{page_title}': {e}")
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
            st.warning(f"Wikipedia infobox failed for '{page_title}' (attempt {_ + 1}/{retries}): {e}")
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
        found_multi_word = False
        for phrase in multi_word_phrases:
            if interest == phrase:
                interest = phrase.replace(" ", "_")
                found_multi_word = True
                break
        processed.append(interest)
    return processed

# --- Playwright Browser Management ---
@st.cache_resource(show_spinner="Initializing Playwright Browser...")
def get_browser():
    # This initializes Playwright and launches a browser.
    # st.cache_resource ensures it's only done once per session.
    pw = sync_playwright().start()
    # Use headless=True for production, False for debugging
    browser = pw.chromium.launch(headless=True)
    return browser

# --- Scraping Functions (Adapted for Playwright) ---
def extract_profile_playwright(page: Page, user_id: str, depth: int) -> dict:
    url = f"https://scholar.google.com/citations?hl=en&user={user_id}"
    page.goto(url, wait_until="domcontentloaded")
    time.sleep(random.uniform(1.5, 3.0)) # Simulate human-like delay

    # Use Playwright's page.locator for robust element finding
    # Name has a unique ID, so it's usually safe
    name_elem = page.locator("#gsc_prf_in")
    name = name_elem.text_content().strip() if name_elem.count() > 0 else "Unknown"

    # Position: Often the first .gsc_prf_il that is NOT #gsc_prf_ivh AND NOT #gsc_prf_int
    position_elem = page.locator(".gsc_prf_il:not(#gsc_prf_ivh):not(#gsc_prf_int)").first
    position = position_elem.text_content().strip() if position_elem.count() > 0 else "Unknown"

    # Email has a specific ID: #gsc_prf_ivh
    email_elem = page.locator("#gsc_prf_ivh")
    email = email_elem.text_content().strip() if email_elem.count() > 0 else "Unknown"

    # Interests has a specific ID: #gsc_prf_int
    interests_elems = page.locator("#gsc_prf_int a") # Your original interests_elems was correct here
    interests_raw = ", ".join(interests_elems.all_text_contents()) if interests_elems.count() > 0 else ""
    
    interest_phrases = normalize_interest_phrases(interests_raw)

    country = infer_country_from_email_field(email)
    institution = get_institution_from_email(email)

    citations_all = "0"
    h_index_all = "0"
    
    # Use Playwright for metrics
    metrics_rows = page.locator("#gsc_rsb_st tbody tr")
    if metrics_rows.count() >= 2:
        try:
            citations_all = metrics_rows.nth(0).locator("td").nth(1).text_content().strip()
            h_index_all = metrics_rows.nth(1).locator("td").nth(1).text_content().strip()
        except Exception:
            pass # Keep default "0"

    # Get coauthors using Playwright
    coauthors = []
    coauthor_links = page.locator(".gsc_rsb_aa .gsc_rsb_a_desc a")
    for i in range(coauthor_links.count()):
        href = coauthor_links.nth(i).get_attribute("href")
        if href and "user=" in href:
            co_id = href.split("user=")[1].split("&")[0]
            coauthors.append(co_id)

    profile = {
        "user_id": user_id,
        "name": name,
        "position": position,
        "email": email,
        "country": country,
        "institution": institution,
        "research_interests": interests_raw,
        "interest_phrases": interest_phrases,
        "citations_all": citations_all,
        "h_index_all": h_index_all,
        "topic_clusters": [], # Populated later
        "search_depth": depth,
        "coauthors": coauthors,
        # Initialize conference participation flags and matched names/institutions
        "Participated_in_ICLR": False,
        "ICLR_Matched_Name": "",
        "ICLR_Institution": "",
        "Participated_in_NeurIPS": False,
        "NeurIPS_Matched_Name": "",
        "NeurIPS_Institution": "",
        "Fuzzy_Matched": False # Initialize this flag
    }
    return profile

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

# --- State Initialization for Streamlit App ---
# Initialize session state variables for persistence across reruns
if 'all_profiles' not in st.session_state:
    st.session_state.all_profiles = []
if 'visited_ids' not in st.session_state:
    st.session_state.visited_ids = set()
if 'crawl_queue' not in st.session_state:
    st.session_state.crawl_queue = deque()
if 'coauthor_graph' not in st.session_state:
    st.session_state.coauthor_graph = nx.Graph()
if 'is_crawling' not in st.session_state:
    st.session_state.is_crawling = False
if 'crawl_status_message' not in st.session_state:
    st.session_state.crawl_status_message = "Idle. Ready to start a new crawl or resume."
if 'crawled_count' not in st.session_state:
    st.session_state.crawled_count = 0
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
if 'iclr_df' not in st.session_state:
    st.session_state.iclr_df = None
if 'neurips_df' not in st.session_state:
    st.session_state.neurips_df = None
if 'fuzzy_cache' not in st.session_state:
    st.session_state.fuzzy_cache = {}


# --- Helper functions for Streamlit integration ---

def save_progress_to_disk():
    # Save profiles DataFrame
    if st.session_state.all_profiles:
        # Ensure all profiles have all EXPECTED_COLUMNS, filling missing with default empty values
        # This prevents issues when creating DataFrame from heterogenous dictionaries
        normalized_profiles = []
        for profile in st.session_state.all_profiles:
            normalized_profile = {col: profile.get(col, "") for col in EXPECTED_COLUMNS}
            # Handle list/set conversion to string for saving
            for list_col in ["interest_phrases", "topic_clusters", "coauthors"]:
                if isinstance(normalized_profile.get(list_col), (list, set)):
                    normalized_profile[list_col] = str(list(normalized_profile[list_col]))
                elif normalized_profile.get(list_col) is None:
                    normalized_profile[list_col] = "[]"
            # Ensure boolean values are saved as readable strings
            for bool_col in ["wiki_deceased", "wiki_is_researcher_ml", "Participated_in_ICLR", "Participated_in_NeurIPS", "Fuzzy_Matched"]:
                if isinstance(normalized_profile.get(bool_col), bool):
                    normalized_profile[bool_col] = str(normalized_profile[bool_col])
                elif normalized_profile.get(bool_col) is None:
                    normalized_profile[bool_col] = "False"

            normalized_profiles.append(normalized_profile)

        df = pd.DataFrame(normalized_profiles, columns=EXPECTED_COLUMNS) # Specify columns to ensure order
        df.to_csv("scholar_profiles.csv", index=False) # Use a simple name for persistence
        st.info(f"💾 Progress saved: {len(st.session_state.all_profiles)} profiles to scholar_profiles.csv")
    
    # Save queue
    with open("queue.txt", "w") as f:
        for item in st.session_state.crawl_queue:
            f.write(json.dumps(item) + "\n")
    st.info("💾 Queue saved to queue.txt")

    # Save graph
    if st.session_state.coauthor_graph.nodes: # Only save if graph is not empty
        nx.write_graphml(st.session_state.coauthor_graph, "coauthor_network.graphml")
        st.info("💾 Co-author network saved to coauthor_network.graphml")
    
    # Save fuzzy cache
    with open("fuzzy_match_cache.json", "w") as f:
        json.dump(st.session_state.fuzzy_cache, f, indent=2)
    st.info("💾 Fuzzy match cache saved to fuzzy_match_cache.json")


def load_previous_state():
    # Load profiles
    if os.path.exists("scholar_profiles.csv"):
        try:
            profiles_df = pd.read_csv("scholar_profiles.csv")
            loaded_profiles = profiles_df.to_dict(orient="records")
            st.session_state.all_profiles = [] # Reset to populate correctly
            st.session_state.visited_ids = set() # Reset visited IDs as well

            for p in loaded_profiles:
                # Ensure fields are correctly parsed, especially lists/dicts stored as strings
                for col in ["interest_phrases", "topic_clusters", "coauthors"]:
                    if isinstance(p.get(col), str):
                        try:
                            p[col] = eval(p[col]) if p[col].startswith('[') else [item.strip() for item in p[col].split(',') if item.strip()]
                        except:
                            p[col] = [] # Fallback to empty list on error
                    elif p.get(col) is None: # Handle None values for lists
                        p[col] = []
                
                # Ensure search_depth is integer
                try:
                    p["search_depth"] = int(float(p.get("search_depth", 0)))
                except (ValueError, TypeError):
                    p["search_depth"] = 0
                
                # Convert boolean strings back to actual booleans
                for bool_col in ["wiki_deceased", "wiki_is_researcher_ml", "Participated_in_ICLR", "Participated_in_NeurIPS", "Fuzzy_Matched"]:
                    if isinstance(p.get(bool_col), str):
                        p[bool_col] = p[bool_col].lower() == 'true'
                    elif p.get(bool_col) is None:
                        p[bool_col] = False

                # Ensure all EXPECTED_COLUMNS are present after loading
                for col in EXPECTED_COLUMNS:
                    if col not in p:
                        p[col] = "" if "institution" in col.lower() or "name" in col.lower() or "summary" in col.lower() or "title" in col.lower() else False if "participated" in col.lower() or "researcher" in col.lower() or "deceased" in col.lower() else [] if "interest" in col.lower() or "coauthor" in col.lower() or "cluster" in col.lower() else "0" if "index" in col.lower() or "citations" in col.lower() else ""

                st.session_state.all_profiles.append(p)
                st.session_state.visited_ids.add(p["user_id"])

            st.success(f"✅ Loaded {len(st.session_state.all_profiles)} profiles from scholar_profiles.csv.")
        except Exception as e:
            st.warning(f"⚠️ Error loading profiles CSV: {e}. Starting with empty profiles.")
            st.session_state.all_profiles = []
            st.session_state.visited_ids = set()

    # Load graph
    if os.path.exists("coauthor_network.graphml"):
        try:
            st.session_state.coauthor_graph = nx.read_graphml("coauthor_network.graphml")
            st.success("✅ Loaded co-author network from coauthor_network.graphml.")
        except Exception as e:
            st.warning(f"⚠️ Error loading graphml file: {e}. Graph file appears corrupted, starting fresh graph.")
            st.session_state.coauthor_graph = nx.Graph()

    # Load queue
    if os.path.exists("queue.txt"):
        try:
            with open("queue.txt", "r") as f:
                lines = [line.strip() for line in f if line.strip()]
            st.session_state.crawl_queue = deque()
            for line in lines:
                try:
                    data = json.loads(line)
                    # Ensure the tuple is (user_id, depth, parent_id)
                    if isinstance(data, list) and len(data) >= 2:
                        st.session_state.crawl_queue.append((data[0], int(data[1]), data[2] if len(data) > 2 else None))
                    else: # Fallback for old/malformed formats
                        st.session_state.crawl_queue.append((str(data), 0, None))
                except json.JSONDecodeError:
                    st.session_state.crawl_queue.append((line, 0, None)) # Fallback for old format (just ID)
            st.success(f"✅ Loaded {len(st.session_state.crawl_queue)} items into the queue from queue.txt.")
        except Exception as e:
            st.warning(f"⚠️ Error loading queue file: {e}. Starting with empty queue.")
            st.session_state.crawl_queue = deque()

    # Load fuzzy cache
    if os.path.exists("fuzzy_match_cache.json"):
        try:
            with open("fuzzy_match_cache.json", "r") as f:
                st.session_state.fuzzy_cache = json.load(f)
            st.success("✅ Loaded fuzzy match cache from fuzzy_match_cache.json.")
        except Exception as e:
            st.warning(f"⚠️ Error loading fuzzy cache: {e}. Starting with empty cache.")
            st.session_state.fuzzy_cache = {}

def enqueue_user(user_id, depth, parent_id=None, prepend=False):
    # Ensure user_id is a string, depth is an int
    user_id = str(user_id)
    depth = int(depth)

    if user_id in st.session_state.visited_ids:
        return
    
    # Check if already in queue to avoid duplicates by user_id
    if any(item[0] == user_id for item in st.session_state.crawl_queue):
        return

    new_item = (user_id, depth, parent_id)
    if prepend:
        st.session_state.crawl_queue.appendleft(new_item)
    else:
        st.session_state.crawl_queue.append(new_item)

def increment_edge_weight(a, b):
    # This now operates on the session_state graph
    if st.session_state.coauthor_graph.has_edge(a, b):
        st.session_state.coauthor_graph[a][b]["weight"] += 1
    else:
        st.session_state.coauthor_graph.add_edge(a, b, weight=1)

def fuzzy_match_conference_participation(profile: dict, conf_name: str, df: pd.DataFrame, name_col='Author', inst_col='Institution', threshold=85):
    """
    Fuzzy matches a profile's name against a conference DataFrame.
    Updates the profile dictionary directly and uses/updates the fuzzy_cache.
    """
    if df is None:
        # st.warning(f"[{conf_name}] Conference data not loaded for fuzzy matching.") # Avoid spamming logs
        return
    
    profile_name = profile.get("name", "").lower()
    if not profile_name:
        profile[f"Participated_in_{conf_name}"] = False
        profile[f"{conf_name}_Matched_Name"] = ""
        profile[f"{conf_name}_Institution"] = ""
        return

    # Check cache first
    cache_key = f"{conf_name}:{profile_name}"
    if cache_key in st.session_state.fuzzy_cache:
        matched_info = st.session_state.fuzzy_cache[cache_key]
        profile[f"Participated_in_{conf_name}"] = matched_info["participated"]
        profile[f"{conf_name}_Matched_Name"] = matched_info["matched_name"]
        profile[f"{conf_name}_Institution"] = matched_info["institution"]
        return

    authors = df[name_col].dropna().unique()
    
    # Process authors for fuzzy matching
    # Create a list of (lower_case_author, original_author, original_institution)
    authors_for_matching = []
    for author_orig in authors:
        inst_orig = ""
        if inst_col in df.columns:
            # Find the institution associated with this specific author entry
            # Assuming one institution per author in the conference data for simplicity
            inst_data = df[df[name_col] == author_orig][inst_col].dropna()
            if not inst_data.empty:
                inst_orig = inst_data.iloc[0]
        authors_for_matching.append((author_orig.lower(), author_orig, inst_orig))

    if not authors_for_matching:
        profile[f"Participated_in_{conf_name}"] = False
        profile[f"{conf_name}_Matched_Name"] = ""
        profile[f"{conf_name}_Institution"] = ""
        st.session_state.fuzzy_cache[cache_key] = {"participated": False, "matched_name": "", "institution": ""}
        return

    # Perform fuzzy match on lowercased names
    best_match_tuple = process.extractOne(profile_name, [a[0] for a in authors_for_matching], scorer=fuzz.token_sort_ratio)
    
    if best_match_tuple and best_match_tuple[1] >= threshold:
        matched_lower_name, score = best_match_tuple
        
        # Find the original author and institution from our prepared list
        original_matched_author = ""
        original_matched_institution = ""
        for lower_name, orig_name, orig_inst in authors_for_matching:
            if lower_name == matched_lower_name:
                original_matched_author = orig_name
                original_matched_institution = orig_inst
                break
        
        profile[f"Participated_in_{conf_name}"] = True
        profile[f"{conf_name}_Matched_Name"] = original_matched_author
        profile[f"{conf_name}_Institution"] = original_matched_institution
        st.success(f"✅ [{conf_name}] Match: {profile['name']} → {original_matched_author} @ {original_matched_institution} (Score: {score})")
        st.session_state.fuzzy_cache[cache_key] = {"participated": True, "matched_name": original_matched_author, "institution": original_matched_institution}
    else:
        profile[f"Participated_in_{conf_name}"] = False
        profile[f"{conf_name}_Matched_Name"] = ""
        profile[f"{conf_name}_Institution"] = ""
        st.info(f"❌ [{conf_name}] No match for {profile['name']} (Best score: {best_match_tuple[1] if best_match_tuple else 0})")
        st.session_state.fuzzy_cache[cache_key] = {"participated": False, "matched_name": "", "institution": ""}


def run_fuzzy_matching_for_all_profiles():
    if st.session_state.iclr_df is None and st.session_state.neurips_df is None:
        st.warning("No conference data uploaded for fuzzy matching.")
        return

    newly_matched_count = 0
    with st.spinner("Running fuzzy matching for conference participation..."):
        # Iterate over a copy to avoid issues if all_profiles is modified during iteration (though unlikely here)
        for profile in st.session_state.all_profiles: # Iterate directly, as modifications are in-place
            # Ensure 'Fuzzy_Matched' key exists and is boolean
            if "Fuzzy_Matched" not in profile:
                profile["Fuzzy_Matched"] = False
            elif isinstance(profile["Fuzzy_Matched"], str): # Convert from string if loaded from CSV
                profile["Fuzzy_Matched"] = profile["Fuzzy_Matched"].lower() == 'true'

            if not profile["Fuzzy_Matched"]:
                fuzzy_match_conference_participation(profile, "ICLR", st.session_state.iclr_df, name_col='Author', inst_col='Institution')
                fuzzy_match_conference_participation(profile, "NeurIPS", st.session_state.neurips_df, name_col='Author', inst_col='Institution')
                profile["Fuzzy_Matched"] = True # Mark as fuzzy-matched
                newly_matched_count += 1
        st.success(f"Fuzzy matching completed. Processed {newly_matched_count} new profiles for fuzzy matching.")
        # Save cache after fuzzy matching all pending profiles
        save_progress_to_disk() # This includes saving fuzzy_match_cache.json


# --- Main Crawling Function (Streamlit-aware) ---
def crawl_bfs_resume_streamlit(browser: BrowserContext, seed_user_ids_input: str, max_crawl_depth: int, max_crawl_seconds: int, save_every: int, fuzzy_run_interval: int, status_placeholder, progress_bar):
    st.session_state.is_crawling = True
    st.session_state.start_time = time.time()
    st.session_state.crawled_count = 0
    profiles_crawled_this_run = 0
    fuzzy_match_counter = 0

    # --- Initial Queue Population ---
    if seed_user_ids_input:
        new_seeds = [id.strip() for id in seed_user_ids_input.split(',') if id.strip()]
        for seed_id in new_seeds:
            # Check if seed_id is already in visited_ids or crawl_queue before adding
            if seed_id not in st.session_state.visited_ids and not any(item[0] == seed_id for item in st.session_state.crawl_queue):
                enqueue_user(seed_id, 0) # Add with depth 0
                status_placeholder.info(f"Added initial seed ID to queue: {seed_id}")
    
    if not st.session_state.crawl_queue:
        status_placeholder.error("Queue is empty and no new seed IDs were provided. Cannot start crawl.")
        st.session_state.is_crawling = False
        st.rerun() # Rerun to update UI status
        return

    # Main crawling loop
    while st.session_state.crawl_queue and st.session_state.is_crawling:
        elapsed_time = time.time() - st.session_state.start_time
        if elapsed_time > max_crawl_seconds:
            status_placeholder.warning(f"🛑 Max crawl time ({max_crawl_seconds}s) reached. Stopping crawl.")
            st.session_state.is_crawling = False
            break
        
        user_id, depth, parent_id = st.session_state.crawl_queue.popleft()
        depth = int(depth) # Ensure depth is an integer

        if user_id in st.session_state.visited_ids:
            continue
        if max_crawl_depth > 0 and depth >= max_crawl_depth: # Changed to >= as depth starts from 0
            status_placeholder.info(f"Skipping {user_id}: Depth {depth} exceeds max_crawl_depth {max_crawl_depth}.")
            continue

        st.session_state.crawled_count += 1
        profiles_crawled_this_run += 1
        current_status_text = f"🔎 Crawling {user_id} at depth {depth} | Queue: {len(st.session_state.crawl_queue)} | Scraped total: {st.session_state.crawled_count}"
        status_placeholder.info(current_status_text)
        # Progress bar based on current crawl count (can be total or per-session)
        progress_bar.progress(min(1.0, st.session_state.crawled_count / (len(st.session_state.visited_ids) + len(st.session_state.crawl_queue) + 1)), text=current_status_text)

        try:
            # Create a new browser context for each profile to isolate sessions
            context = browser.new_context()
            page = context.new_page() 
            
            profile = extract_profile_playwright(page, user_id, depth)
            
            page.close() # Close page after use
            context.close() # Close context as well
            
            st.session_state.visited_ids.add(user_id)
            
            # --- Wikipedia Info Retrieval ---
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

            # Add co-author relationships to the graph
            for co_id in profile.get("coauthors", []):
                if co_id != user_id: # Avoid self-loops
                    increment_edge_weight(user_id, co_id)
                    # Enqueue co-authors for further crawling if within depth limit
                    if depth + 1 <= max_crawl_depth: # Ensure co-authors don't exceed max depth
                        enqueue_user(co_id, depth + 1, parent_id=user_id)

            st.session_state.all_profiles.append(profile)

            # --- Periodic Saves and Fuzzy Matching ---
            if profiles_crawled_this_run % save_every == 0:
                save_progress_to_disk()
            
            # Run fuzzy matching periodically
            if profiles_crawled_this_run % fuzzy_run_interval == 0:
                run_fuzzy_matching_for_all_profiles()

        except Exception as e:
            status_placeholder.error(f"❌ Error crawling {user_id}: {e}")
            # Consider requeueing or logging the failed ID for retry
            # For now, just skip and move on
            
        time.sleep(random.uniform(0.5, 2.0)) # Politeness delay between profiles

    st.session_state.is_crawling = False
    save_progress_to_disk() # Final save
    run_fuzzy_matching_for_all_profiles() # Final fuzzy match
    status_placeholder.success("✅ Crawl finished (or stopped).")
    st.rerun() # Rerun to update UI after crawl stops

# --- Streamlit UI Layout ---
st.set_page_config(layout="wide", page_title="Scholar Profile Crawler", page_icon="🔍")

st.title("Scholar Profile Crawler 🕸️")

st.markdown("""
This application allows you to crawl Google Scholar profiles and extract public information,
including co-authors, research interests, and integrate with Wikipedia data.
""")

### Configuration
col1, col2, col3 = st.columns(3)

with col1:
    seed_user_ids_input = st.text_area(
        "Enter Seed Google Scholar User IDs (comma-separated):",
        help="Example: 'N-p-Y1QAAAAJ, 39P0t4YAAAAJ'",
        key="seed_ids_input"
    )
    max_crawl_depth = st.number_input(
        "Max Crawl Depth (0 for seeds only, 1 for seeds + their direct co-authors, etc.):",
        min_value=0, value=1, step=1, key="max_depth"
    )

with col2:
    max_crawl_seconds = st.number_input(
        "Max Crawl Time (seconds):",
        min_value=60, value=3600, step=60, key="max_time"
    )
    save_every = st.number_input(
        "Save Progress Every (profiles crawled):",
        min_value=1, value=10, step=5, key="save_freq"
    )

with col3:
    fuzzy_run_interval = st.number_input(
        "Run Fuzzy Match Every (profiles crawled):",
        min_value=1, value=50, step=10, key="fuzzy_freq"
    )
    st.markdown("---")
    st.subheader("Conference Data Upload (Optional)")
    iclr_file = st.file_uploader("Upload ICLR Authors CSV", type=["csv"], key="iclr_upload")
    neurips_file = st.file_uploader("Upload NeurIPS Authors CSV", type=["csv"], key="neurips_upload")

if iclr_file:
    try:
        st.session_state.iclr_df = pd.read_csv(iclr_file)
        st.success("ICLR data loaded successfully!")
    except Exception as e:
        st.error(f"Error loading ICLR CSV: {e}")
if neurips_file:
    try:
        st.session_state.neurips_df = pd.read_csv(neurips_file)
        st.success("NeurIPS data loaded successfully!")
    except Exception as e:
        st.error(f"Error loading NeurIPS CSV: {e}")


### Control Panel
status_placeholder = st.empty()
progress_bar = st.progress(0, text="Initializing...")

col_buttons1, col_buttons2, col_buttons3 = st.columns(3)

if col_buttons1.button("Start Crawl", type="primary", disabled=st.session_state.is_crawling):
    if not st.session_state.crawl_queue and not seed_user_ids_input:
        status_placeholder.error("Please provide seed user IDs or load a queue to start crawling.")
    else:
        st.session_state.is_crawling = True
        status_placeholder.info("Starting crawl...")
        with st.spinner("Crawling in progress..."):
            try:
                browser = get_browser() # Get cached browser instance
                crawl_bfs_resume_streamlit(
                    browser,
                    seed_user_ids_input,
                    max_crawl_depth,
                    max_crawl_seconds,
                    save_every,
                    fuzzy_run_interval,
                    status_placeholder,
                    progress_bar
                )
            except Exception as e:
                status_placeholder.error(f"An error occurred during crawl: {e}")
                st.session_state.is_crawling = False
                st.rerun() # Rerun to update button state

if col_buttons2.button("Stop Crawl", disabled=not st.session_state.is_crawling):
    st.session_state.is_crawling = False
    status_placeholder.warning("Crawl stopping... Please wait for current operation to complete.")
    st.rerun() # Trigger a rerun to allow the loop to break

if col_buttons3.button("Load Previous State", disabled=st.session_state.is_crawling):
    load_previous_state()
    st.rerun() # Rerun to update display with loaded data

if st.button("Run Fuzzy Matching on Loaded Data"):
    run_fuzzy_matching_for_all_profiles()
    st.rerun() # Rerun to refresh the data display

### Current State
st.subheader("Crawling Statistics")
st.write(f"**Status:** {st.session_state.crawl_status_message}")
st.write(f"**Profiles Scraped (Total):** {len(st.session_state.all_profiles)}")
st.write(f"**Profiles in Queue:** {len(st.session_state.crawl_queue)}")
st.write(f"**Unique Visited IDs:** {len(st.session_state.visited_ids)}")

if st.session_state.start_time:
    current_elapsed_time = time.time() - st.session_state.start_time
    st.write(f"**Elapsed Crawl Time:** {int(current_elapsed_time)} seconds")


# Filter and display data
st.subheader("Scraped Profiles")

if st.session_state.all_profiles:
    # Ensure all columns are present before creating DataFrame for consistent display
    display_profiles = []
    for profile in st.session_state.all_profiles:
        temp_profile = {col: profile.get(col, None) for col in EXPECTED_COLUMNS}
        display_profiles.append(temp_profile)

    df_display = pd.DataFrame(display_profiles)
    
    # Optionally, convert list-like columns for better display
    for col in ["interest_phrases", "topic_clusters", "coauthors"]:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(lambda x: ", ".join(x) if isinstance(x, (list, set)) else x)

    st.dataframe(df_display)

    csv_output = df_display.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Scraped Data as CSV",
        data=csv_output,
        file_name="scholar_profiles_crawled.csv",
        mime="text/csv",
    )
else:
    st.info("No profiles scraped yet. Start a crawl or load previous state.")

st.subheader("Co-author Network (Nodes and Edges)")
if st.session_state.coauthor_graph.nodes:
    st.write(f"**Number of nodes:** {st.session_state.coauthor_graph.number_of_nodes()}")
    st.write(f"**Number of edges:** {st.session_state.coauthor_graph.number_of_edges()}")
    
    # Display top 10 co-author relationships by weight
    if st.session_state.coauthor_graph.edges:
        edges_with_weights = []
        for u, v, data in st.session_state.coauthor_graph.edges(data=True):
            edges_with_weights.append({'Source': u, 'Target': v, 'Weight': data.get('weight', 1)})
        edges_df = pd.DataFrame(edges_with_weights)
        st.write("Top 10 Co-author Relationships (by weight):")
        st.dataframe(edges_df.sort_values(by='Weight', ascending=False).head(10))
else:
    st.info("No co-author network data available yet.")

st.markdown("---")
st.markdown("Built with ❤️ by Your Name/Team")
