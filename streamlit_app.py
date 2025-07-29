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

# === CONFIG ===
# These will be configurable via Streamlit UI
# PROGRESS_CSV = "scholar_profiles_progressssss.csv" # Managed by session state/disk
# GRAPH_GRAPHML = "coauthor_network_progressssss.graphml" # Managed by session state/disk
# QUEUE_FILE = "queue.txt" # Managed by session state/disk
# FUZZY_CACHE_PATH = "fuzzy_match_cache.json" # Managed by session state/disk

EXPECTED_COLUMNS = [
    "user_id", "name", "position", "email", "country", "institution", "research_interests",
    "interest_phrases", "citations_all", "h_index_all", "topic_clusters", "search_depth",
    "Participated_in_ICLR", "ICLR_Matched_Name", "wiki_birth_name", "wiki_name",
    "wiki_birth_date", "wiki_birth_place", "wiki_death_date", "wiki_death_place",
    "wiki_fields", "wiki_work_institution", "wiki_alma_mater", "wiki_notable_students",
    "wiki_thesis_title", "wiki_thesis_year", "wiki_thesis_url", "wiki_known_for",
    "wiki_awards", "wiki_deceased", "wiki_wiki_summary", "wiki_is_researcher_ml",
    "wiki_matched_title", "coauthors", "Fuzzy_Matched", "ICLR_Institution",
    "Participated_in_NeurIPS", "NeurIPS_Institution"
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
from playwright.sync_api import sync_playwright

@st.cache_resource(show_spinner="Initializing Playwright Browser...")
def get_browser():
    # This initializes Playwright and launches a browser.
    # st.cache_resource ensures it's only done once per session.
    pw = sync_playwright().start()
    browser = pw.chromium.launch(headless=True) # Set headless=False for visual debugging if needed
    return browser

# --- Scraping Functions (Adapted for Playwright) ---
def extract_profile_playwright(page, user_id, depth):
    url = f"https://scholar.google.com/citations?hl=en&user={user_id}"
    page.goto(url, wait_until="domcontentloaded")
    time.sleep(random.uniform(1.5, 3.0)) # Simulate human-like delay

    # Use Playwright's page.locator for robust element finding
    # Name has a unique ID, so it's usually safe
    name_elem = page.locator("#gsc_prf_in")
    name = name_elem.text_content().strip() if name_elem.count() > 0 else "Unknown"

    # --- FIX START ---

    # Original problem: ".gsc_prf_il" is too general.
    # The 'position' element is typically the first or second div with that class
    # and *doesn't* have an 'id'.

    # Strategy 1: Target the specific parent or sibling relationship.
    # On Google Scholar, the position usually comes right after the name and doesn't have an ID
    # while the email and interests DO have IDs.
    
    # Try to find the .gsc_prf_il that is *not* the email or interests
    # Or more directly, the first .gsc_prf_il after #gsc_prf_in that doesn't have an ID
    # A robust way is to use XPath or a more specific CSS selector.
    
    # Let's target the element after #gsc_prf_in that is *not* #gsc_prf_ivh or #gsc_prf_int
    # Or, given the structure, often the first `gsc_prf_il` *without* an ID is the position.
    
    # Method 1: Use specific IDs where available, and be smart for others
    # Position: Often the first .gsc_prf_il that is NOT #gsc_prf_ivh AND NOT #gsc_prf_int
    position_elem = page.locator(".gsc_prf_il:not(#gsc_prf_ivh):not(#gsc_prf_int)").first
    position = position_elem.text_content().strip() if position_elem.count() > 0 else "Unknown"

    # Email has a specific ID: #gsc_prf_ivh
    email_elem = page.locator("#gsc_prf_ivh")
    email = email_elem.text_content().strip() if email_elem.count() > 0 else "Unknown"

    # Interests has a specific ID: #gsc_prf_int
    interests_elems = page.locator("#gsc_prf_int a") # Your original interests_elems was correct here
    interests_raw = ", ".join(interests_elems.all_text_contents()) if interests_elems.count() > 0 else ""
    
    # --- FIX END ---

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
        df = pd.DataFrame(st.session_state.all_profiles)
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
            st.session_state.all_profiles = profiles_df.to_dict(orient="records")
            for p in st.session_state.all_profiles:
                # Ensure fields are correctly parsed, especially lists/dicts stored as strings
                for col in ["interest_phrases", "topic_clusters", "coauthors"]:
                    if isinstance(p.get(col), str):
                        try:
                            p[col] = eval(p[col]) if p[col].startswith('[') else p[col].split(', ') # Handle lists
                        except:
                            p[col] = []
                try:
                    p["search_depth"] = int(float(p.get("search_depth", 0)))
                except (ValueError, TypeError):
                    p["search_depth"] = 0
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
                    st.session_state.crawl_queue.append(tuple(data))
                except json.JSONDecodeError:
                    st.session_state.crawl_queue.append((line, 0, None)) # Fallback for old format
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
    if user_id in st.session_state.visited_ids:
        return
    if any(user_id == item[0] for item in st.session_state.crawl_queue):
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

def fuzzy_match_conference_participation(profile, conf_name, df, name_col='Author', inst_col='Institution', threshold=85):
    if df is None:
        # st.warning(f"[{conf_name}] Conference data not loaded for fuzzy matching.") # Avoid spamming logs
        return
    authors = df[name_col].dropna().unique()
    authors_lower = [a.lower() for a in authors]

    profile_name = profile.get("name", "").lower()
    if not profile_name:
        profile[f"Participated_in_{conf_name}"] = False
        profile[f"{conf_name}_Institution"] = ""
        return

    # Check cache first
    cache_key = f"{conf_name}:{profile_name}"
    if cache_key in st.session_state.fuzzy_cache:
        matched_info = st.session_state.fuzzy_cache[cache_key]
        profile[f"Participated_in_{conf_name}"] = matched_info["participated"]
        profile[f"{conf_name}_Institution"] = matched_info["institution"]
        # st.success(f"✅ [{conf_name}] Match from cache for {profile['name']}") # For debugging
        return

    match, score = process.extractOne(profile_name, authors_lower, scorer=fuzz.token_sort_ratio)
    if score >= threshold:
        match_idx = authors_lower.index(match)
        matched_author = authors[match_idx]
        matched_institution = df[df[name_col] == matched_author][inst_col].dropna().iloc[0] if inst_col in df.columns else ""
        
        profile[f"Participated_in_{conf_name}"] = True
        profile[f"{conf_name}_Institution"] = matched_institution
        st.success(f"✅ [{conf_name}] Match: {profile['name']} → {matched_author} @ {matched_institution}")
        
        st.session_state.fuzzy_cache[cache_key] = {"participated": True, "institution": matched_institution}
    else:
        profile[f"Participated_in_{conf_name}"] = False
        profile[f"{conf_name}_Institution"] = ""
        st.info(f"❌ [{conf_name}] No match for {profile['name']}")
        st.session_state.fuzzy_cache[cache_key] = {"participated": False, "institution": ""}


def run_fuzzy_matching_for_all_profiles():
    if st.session_state.iclr_df is None and st.session_state.neurips_df is None:
        st.warning("No conference data uploaded for fuzzy matching.")
        return

    newly_matched_count = 0
    with st.spinner("Running fuzzy matching for conference participation..."):
        for profile in st.session_state.all_profiles:
            # Add a 'Fuzzy_Matched' key to each profile if it doesn't exist
            if "Fuzzy_Matched" not in profile:
                profile["Fuzzy_Matched"] = False

            if not profile["Fuzzy_Matched"]:
                fuzzy_match_conference_participation(profile, "ICLR", st.session_state.iclr_df)
                fuzzy_match_conference_participation(profile, "NeurIPS", st.session_state.neurips_df)
                profile["Fuzzy_Matched"] = True
                newly_matched_count += 1
        st.success(f"Fuzzy matching completed. Processed {newly_matched_count} new profiles for fuzzy matching.")
        # Save cache after fuzzy matching all pending profiles
        save_progress_to_disk() # This includes saving fuzzy_match_cache.json


# --- Main Crawling Function (Streamlit-aware) ---
def crawl_bfs_resume_streamlit(browser_page, seed_user_id, max_crawl_depth, max_crawl_seconds, save_every, fuzzy_run_interval, status_placeholder, progress_bar):
    st.session_state.is_crawling = True
    st.session_state.start_time = time.time()
    st.session_state.crawled_count = 0
    new_profiles_this_run = 0

    if not st.session_state.crawl_queue and seed_user_id:
        enqueue_user(seed_user_id, 0)
        status_placeholder.info(f"Queue initialized with seed: {seed_user_id}")
    elif not st.session_state.crawl_queue:
        status_placeholder.error("Queue is empty and no seed ID provided. Cannot start crawl.")
        st.session_state.is_crawling = False
        return

    while st.session_state.crawl_queue and st.session_state.is_crawling:
        # Check if time limit is exceeded
        elapsed_time = time.time() - st.session_state.start_time
        if elapsed_time > max_crawl_seconds:
            status_placeholder.warning(f"🛑 Max crawl time ({max_crawl_seconds}s) reached. Stopping crawl.")
            st.session_state.is_crawling = False
            break
        
        # Check for Stop button press (Streamlit reruns on interaction)
        if not st.session_state.is_crawling: # This gets updated if user clicks Stop
            status_placeholder.info("Crawl manually stopped.")
            break

        user_id, depth, parent_id = st.session_state.crawl_queue.popleft()
        depth = int(depth) # Ensure depth is an integer

        # Skip if already visited or depth limit exceeded
        if user_id in st.session_state.visited_ids:
            # status_placeholder.info(f"Skipping already visited user {user_id}") # Too verbose for status
            continue
        if max_crawl_depth > 0 and depth > max_crawl_depth:
            status_placeholder.info(f"Skipping {user_id}: Depth {depth} exceeds max_crawl_depth {max_crawl_depth}.")
            continue

        status_placeholder.info(f"🔎 Crawling {user_id} at depth {depth} (Queue: {len(st.session_state.crawl_queue)}, Scraped: {st.session_state.crawled_count})")
        progress_bar.progress((st.session_state.crawled_count % 100) / 100.0, text=f"Crawled: {st.session_state.crawled_count} | Queue: {len(st.session_state.crawl_queue)} | Current: {user_id}")

        try:
            # Use Playwright page from the cached browser
            browser_instance = get_browser()
            page = browser_instance.new_page() # Create a new page for each scrape
            profile = extract_profile_playwright(page, user_id, depth)
            page.close() # Close page after use
            
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

            st.session_state.all_profiles.append(profile)
            new_profiles_this_run += 1
            st.session_state.crawled_count += 1

            status_placeholder.success(f"✅ Scraped: {profile['name']} (h-index: {profile['h_index_all']}, {profile['country']}, depth: {depth})")

            # Enqueue co-authors (limit 20)
            for co_id in profile.get("coauthors", [])[:20]:
                enqueue_user(co_id, depth + 1, user_id)
                # Add edge to graph
                if user_id != co_id: # Avoid self-loops
                    increment_edge_weight(user_id, co_id)


            # Save progress and run fuzzy matching periodically
            if st.session_state.crawled_count % save_every == 0:
                save_progress_to_disk()
                status_placeholder.info(f"💾 Saved progress after {st.session_state.crawled_count} profiles.")
                
            if st.session_state.crawled_count % fuzzy_run_interval == 0:
                 # Run fuzzy matching on all profiles to catch newly added ones
                run_fuzzy_matching_for_all_profiles()
                st.info("Fuzzy matching completed for current batch.")

        except Exception as e:
            status_placeholder.error(f"❌ Error scraping {user_id}: {e}")
            # Consider re-enqueuing with a higher depth or blacklisting
            # For now, just skip to next
            st.session_state.visited_ids.add(user_id) # Mark as visited to avoid re-attempting immediately
            time.sleep(random.uniform(5, 10)) # Longer delay on error

    st.session_state.is_crawling = False
    save_progress_to_disk() # Final save
    status_placeholder.success(f"✅ Crawl finished. Total new profiles scraped in this run: {new_profiles_this_run}. Total profiles in data: {len(st.session_state.all_profiles)}.")
    st.session_state.crawl_status_message = f"✅ Crawl finished. {new_profiles_this_run} new profiles scraped."
    st.experimental_rerun() # Rerun to update final status


# --- Streamlit UI Layout ---

st.set_page_config(layout="wide", page_title="Scholar Scraper App")
st.title("👨‍💻 Scholar Scraper & Network Analyzer")

# --- Configuration Section ---
st.header("⚙️ Configuration")

col1, col2 = st.columns(2)

with col1:
    SEED_USER_ID = st.text_input(
        "Start Crawl from Scholar User ID (e.g., `0b_Q5gcAAAAJ`)",
        value="0b_Q5gcAAAAJ"
    )
    MAX_CRAWL_DEPTH = st.number_input(
        "Maximum Crawl Depth (0 for no limit, BE CAREFUL)",
        min_value=0,
        value=3, # Adjusted to a more reasonable default
        help="How many 'hops' away from the seed user to crawl (co-authors of co-authors, etc.). High values can lead to very long crawls."
    )
    SAVE_EVERY = st.number_input(
        "Save Progress Every (profiles)",
        min_value=1,
        value=5,
        help="Number of profiles scraped before saving data to disk."
    )
    
with col2:
    MAX_CRAWL_SECONDS = st.number_input(
        "Maximum Crawl Time (seconds)",
        min_value=60, # Minimum 1 minute
        value=3600, # Default 1 hour
        help="Maximum time (in seconds) the crawl will run. Set to a reasonable value to prevent indefinite running."
    )
    FUZZY_RUN_INTERVAL = st.number_input(
        "Run Fuzzy Match Every (profiles)",
        min_value=1,
        value=5,
        help="Number of profiles scraped before re-running fuzzy matching for conference participation."
    )

# --- Conference Data Upload ---
st.subheader("⬆️ Conference Data Upload (Optional)")
st.info("Upload .parquet files for ICLR and NeurIPS to enable fuzzy matching for conference participation.")

col_iclr, col_neurips = st.columns(2)

with col_iclr:
    uploaded_iclr_file = st.file_uploader(
        "Upload ICLR Parquet (e.g., `iclr_2020_2025_combined_data.parquet`)",
        type=["parquet"],
        accept_multiple_files=False,
        key="iclr_uploader"
    )
    if uploaded_iclr_file is not None:
        try:
            st.session_state.iclr_df = pd.read_parquet(uploaded_iclr_file)
            st.success(f"ICLR data loaded: {len(st.session_state.iclr_df)} rows.")
        except Exception as e:
            st.error(f"Error loading ICLR Parquet: {e}")
            st.session_state.iclr_df = None
    elif st.session_state.iclr_df is not None:
        st.success("ICLR data previously loaded.")

with col_neurips:
    uploaded_neurips_file = st.file_uploader(
        "Upload NeurIPS Parquet (e.g., `neurips_2020_2024_combined_data.parquet`)",
        type=["parquet"],
        accept_multiple_files=False,
        key="neurips_uploader"
    )
    if uploaded_neurips_file is not None:
        try:
            st.session_state.neurips_df = pd.read_parquet(uploaded_neurips_file)
            st.success(f"NeurIPS data loaded: {len(st.session_state.neurips_df)} rows.")
        except Exception as e:
            st.error(f"Error loading NeurIPS Parquet: {e}")
            st.session_state.neurips_df = None
    elif st.session_state.neurips_df is not None:
        st.success("NeurIPS data previously loaded.")

## Actions

st.header("⚡ Actions")

status_placeholder = st.empty()
progress_bar = st.progress(0, text="Crawling progress...")

col_actions_1, col_actions_2, col_actions_3 = st.columns(3)

with col_actions_1:
    if st.button("▶️ Start Crawl", disabled=st.session_state.is_crawling):
        if not SEED_USER_ID:
            status_placeholder.error("Please provide a seed user ID to start the crawl.")
        else:
            status_placeholder.info("Starting crawl...")
            # This triggers the crawl function, which will update session_state
            crawl_bfs_resume_streamlit(
                get_browser(), # Pass the cached Playwright browser instance
                SEED_USER_ID,
                MAX_CRAWL_DEPTH,
                MAX_CRAWL_SECONDS,
                SAVE_EVERY,
                FUZZY_RUN_INTERVAL,
                status_placeholder,
                progress_bar
            )
            # st.session_state.is_crawling will be False after crawl_bfs_resume_streamlit returns
            st.experimental_rerun() # Rerun to update UI after crawl finishes/stops

with col_actions_2:
    if st.button("⏹️ Stop Crawl", disabled=not st.session_state.is_crawling):
        st.session_state.is_crawling = False
        status_placeholder.warning("Attempting to stop crawl...")
        # The crawl_bfs_resume_streamlit loop will check st.session_state.is_crawling and break

with col_actions_3:
    if st.button("🧹 Clear All Scraped Data & Cache"):
        if os.path.exists("scholar_profiles.csv"): os.remove("scholar_profiles.csv")
        if os.path.exists("queue.txt"): os.remove("queue.txt")
        if os.path.exists("coauthor_network.graphml"): os.remove("coauthor_network.graphml")
        if os.path.exists("fuzzy_match_cache.json"): os.remove("fuzzy_match_cache.json")
        
        # Reset session state variables
        st.session_state.all_profiles = []
        st.session_state.visited_ids = set()
        st.session_state.crawl_queue = deque()
        st.session_state.coauthor_graph = nx.Graph()
        st.session_state.fuzzy_cache = {}
        st.session_state.crawled_count = 0
        st.session_state.is_crawling = False
        st.session_state.crawl_status_message = "All data cleared. Ready for a fresh start."
        st.success("All scraped data and cache cleared.")
        st.experimental_rerun()

# Load state on first run or after clearing
if st.session_state.crawled_count == 0 and not st.session_state.all_profiles: # Only load if empty
    load_previous_state()

## Current Status

st.header("📊 Current Status")
st.write(f"**Crawl Status:** {st.session_state.crawl_status_message}")
st.write(f"**Profiles Scraped:** {len(st.session_state.all_profiles)}")
st.write(f"**Queue Size:** {len(st.session_state.crawl_queue)}")
st.write(f"**Unique Visited Profiles:** {len(st.session_state.visited_ids)}")

# Display Scraped Data (optional, use st.expander for large data)
if st.session_state.all_profiles:
    st.subheader("Latest Scraped Profiles")
    st.dataframe(pd.DataFrame(st.session_state.all_profiles).tail(10)) # Show last 10
    
    st.subheader("Full Scraped Data")
    csv_data = pd.DataFrame(st.session_state.all_profiles).to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Scraped Data as CSV",
        data=csv_data,
        file_name="scholar_profiles_full.csv",
        mime="text/csv"
    )

    # You could add more visualizations here if desired, e.g., graph analysis
