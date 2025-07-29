import os
import gc
import psutil
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import random
import json
import re
import pandas as pd
import tldextract
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import networkx as nx
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
from fuzzywuzzy import process, fuzz
from collections import Counter, defaultdict, deque
from contextlib import contextmanager
import logging
from datetime import datetime
import threading
import concurrent.futures
from functools import lru_cache
import pickle

import requests
import mwparserfromhell
import difflib
from transformers import pipeline
import torch

# Enhanced logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Memory monitoring
def log_memory_usage(label=""):
    process = psutil.Process()
    memory_info = process.memory_info()
    logger.info(f"Memory {label}: RSS={memory_info.rss / 1024 / 1024:.1f}MB, VMS={memory_info.vms / 1024 / 1024:.1f}MB")

# Optimized device setup with memory management
@lru_cache(maxsize=1)
def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

# Lazy loading of classifier to save memory
_classifier = None
def get_classifier():
    global _classifier
    if _classifier is None:
        device = get_device()
        _classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=0 if device in ["cuda", "mps"] else -1
        )
        # Clear cache after loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return _classifier

# Optimized constants with smaller memory footprint
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

# Session management for HTTP requests
class SessionManager:
    def __init__(self):
        self.session = None
        self.last_used = 0
        self.timeout = 300  # 5 minutes
        
    def get_session(self):
        current_time = time.time()
        if self.session is None or (current_time - self.last_used) > self.timeout:
            if self.session:
                self.session.close()
            self.session = requests.Session()
            self.session.headers.update({'User-Agent': 'Mozilla/5.0 (compatible; WikiInfoBot/1.0)'})
        self.last_used = current_time
        return self.session
    
    def close(self):
        if self.session:
            self.session.close()
            self.session = None

session_manager = SessionManager()

def clean_wiki_markup(raw_text):
    """Optimized wiki markup cleaning"""
    if not raw_text or len(raw_text) > 10000:  # Skip very large texts
        return ""
        
    try:
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
    except Exception:
        return str(raw_text)[:500]  # Fallback to truncated text

@lru_cache(maxsize=1000)
def fuzzy_wikipedia_search(name, threshold=0.90, max_results=5):
    """Cached Wikipedia search"""
    if not name or len(name) < 3:
        return None
        
    session = session_manager.get_session()
    URL = "https://en.wikipedia.org/w/api.php"
    PARAMS = {"action": "query", "list": "search", "srsearch": name, "srlimit": max_results, "format": "json"}
    
    try:
        response = session.get(url=URL, params=PARAMS, timeout=10)
        if response.status_code == 200:
            data = response.json()
            search_results = data.get("query", {}).get("search", [])
            best_match, best_score = None, 0
            for result in search_results:
                title = result['title']
                score = difflib.SequenceMatcher(None, name.lower(), title.lower()).ratio()
                if score > best_score:
                    best_score, best_match = score, title
            return best_match if best_score >= threshold else None
    except Exception as e:
        logger.warning(f"Wikipedia search failed for {name}: {e}")
        return None

@lru_cache(maxsize=500)
def get_wikipedia_summary(page_title):
    """Cached Wikipedia summary retrieval"""
    if not page_title:
        return ""
        
    session = session_manager.get_session()
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{page_title}"
    
    try:
        response = session.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get("extract", "")[:1000]  # Limit summary length
    except Exception as e:
        logger.warning(f"Wikipedia summary failed for {page_title}: {e}")
    return ""

def get_selected_infobox_fields(page_title, fields_to_extract):
    """Optimized infobox extraction"""
    if not page_title:
        return {k: "" for k in fields_to_extract}, None
        
    session = session_manager.get_session()
    URL = "https://en.wikipedia.org/w/api.php"
    PARAMS = {"action": "query", "format": "json", "titles": page_title,
              "prop": "revisions", "rvprop": "content", "rvslots": "main"}
    
    try:
        response = session.get(url=URL, params=PARAMS, timeout=15)
        if response.status_code == 200:
            data = response.json()
            page = next(iter(data['query']['pages'].values()))
            if "missing" in page or 'revisions' not in page:
                return {k: "" for k in fields_to_extract}, None
            
            matched_title = page.get('title', page_title)
            wikitext = page['revisions'][0]['slots']['main']['*']
            
            # Limit wikitext size to prevent memory issues
            if len(wikitext) > 50000:
                wikitext = wikitext[:50000]
            
            wikicode = mwparserfromhell.parse(wikitext)
            infobox = next((t for t in wikicode.filter_templates() if t.name.lower().strip().startswith("infobox")), None)
            extracted = {}
            if infobox:
                for key in fields_to_extract:
                    try:
                        val = infobox.get(key).value.strip()
                        extracted[key] = clean_wiki_markup(str(val))[:500]  # Limit field length
                    except Exception:
                        extracted[key] = ""
            else:
                extracted = {k: "" for k in fields_to_extract}
            return extracted, matched_title
    except Exception as e:
        logger.warning(f"Infobox extraction failed for {page_title}: {e}")
    
    return {k: "" for k in fields_to_extract}, None

def classify_summary(summary):
    """Optimized classification with batch processing support"""
    summary = summary.strip()
    if not summary or len(summary) < 10:
        return False
    
    # Truncate very long summaries
    if len(summary) > 1000:
        summary = summary[:1000]
    
    try:
        classifier = get_classifier()
        classification = classifier(summary, candidate_labels)
        top_label = classification['labels'][0].lower()
        return top_label in [label.lower() for label in researcher_labels]
    except Exception as e:
        logger.warning(f"Classification failed: {e}")
        return False

# Cache for author info
author_info_cache = {}

def get_author_wikipedia_info(author_name):
    """
    Cached author information retrieval
    """
    if not author_name or author_name in author_info_cache:
        return author_info_cache.get(author_name, {})
    
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
        author_info_cache[author_name] = info
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

    author_info_cache[author_name] = info
    return info

# === OPTIMIZED CONFIG ===
SAVE_EVERY = 3  # Save more frequently for GCE
MAX_CRAWL_DEPTH = 800000000000
SEED_USER_ID = "0b_Q5gcAAAAJ"
PROGRESS_CSV = "scholar_profiles_progressssss.csv"
GRAPH_GRAPHML = "coauthor_network_progressssss.graphml"
QUEUE_FILE = "queue.txt"
NUM_PHRASE_CLUSTERS = 20
MAX_CRAWL_SECONDS = 3600000
BATCH_SIZE = 10  # Process in smaller batches
MAX_MEMORY_MB = 3000  # Memory limit for GCE

EXPECTED_COLUMNS = [
    "user_id", "name", "position", "email", "country", "institution", "research_interests",
    "interest_phrases", "citations_all", "h_index_all", "topic_clusters", "search_depth",
    "Participated_in_ICLR", "wiki_birth_name", "wiki_name",
    "wiki_birth_date", "wiki_birth_place", "wiki_death_date", "wiki_death_place",
    "wiki_fields", "wiki_work_institution", "wiki_alma_mater", "wiki_notable_students",
    "wiki_thesis_title", "wiki_thesis_year", "wiki_thesis_url", "wiki_known_for",
    "wiki_awards", "wiki_deceased", "wiki_wiki_summary", "wiki_is_researcher_ml",
    "wiki_matched_title", "coauthors", "Fuzzy_Matched", "ICLR_Institution",
    "Participated_in_NeurIPS", "NeurIPS_Institution"
]

def ensure_progress_csv(path):
    """Optimized CSV initialization with backward compatibility"""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        logger.info("🆕 Initializing empty progress CSV with headers...")
        pd.DataFrame(columns=EXPECTED_COLUMNS).to_csv(path, index=False)
    else:
        try:
            # Read only the header to check columns
            df_header = pd.read_csv(path, nrows=0)
            missing_cols = [col for col in EXPECTED_COLUMNS if col not in df_header.columns]
            extra_cols = [col for col in df_header.columns if col not in EXPECTED_COLUMNS]
            
            if missing_cols:
                logger.warning(f"⚠️ CSV missing columns: {missing_cols}")
                # Instead of reinitializing, add missing columns
                df = pd.read_csv(path)
                for col in missing_cols:
                    df[col] = ""  # Add missing columns with empty values
                df.to_csv(path, index=False)
                logger.info(f"✅ Added missing columns to existing CSV")
            
            if extra_cols:
                logger.info(f"ℹ️ CSV has extra columns that will be preserved: {extra_cols}")
                
        except Exception as e:
            logger.error(f"❌ Error reading progress CSV: {e} — reinitializing.")
            pd.DataFrame(columns=EXPECTED_COLUMNS).to_csv(path, index=False)

# Optimized mappings with reduced memory footprint
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
}

suffix_country_map = {
    'edu.sg': 'sg', 'gov.sg': 'sg', 'ac.uk': 'uk', 'edu.au': 'au', 'edu.cn': 'cn',
    'edu.in': 'in', 'edu.ca': 'ca', 'edu': 'us', 'gov': 'us', 'ac.jp': 'jp',
    'ac.kr': 'kr', 'ac.at': 'at', 'ac.be': 'be', 'ac.nz': 'nz', 'com': 'unknown',
    'org': 'unknown', 'net': 'unknown', 'sg': 'sg', 'uk': 'uk', 'us': 'us',
    'fr': 'fr', 'de': 'de', 'at': 'at', 'ca': 'ca', 'au': 'au', 'cn': 'cn',
    'jp': 'jp', 'kr': 'kr',
}

synonym_map = {"ml": "machine learning", "ai": "artificial intelligence"}

multi_word_phrases = [
    "machine learning", "artificial intelligence", "quantum chemistry",
    "computational materials science", "deep learning", "molecular dynamics",
    "homogeneous catalysis", "organometallic chemistry", "polymer chemistry",
    "drug discovery", "genome engineering", "synthetic biology", "protein engineering",
    "metabolic engineering", "quantum computing", "density functional theory"
]

@lru_cache(maxsize=1000)
def normalize_interest_phrases(raw_text):
    """Cached interest phrase normalization"""
    if not raw_text:
        return []
    interests = [s.strip().lower() for s in raw_text.split(",") if s.strip()]
    processed = []
    for interest in interests:
        interest = synonym_map.get(interest, interest)
        for phrase in multi_word_phrases:
            if interest == phrase:
                interest = phrase.replace(" ", "_")
        processed.append(interest)
    return processed

# Memory-aware driver management
@contextmanager
def get_driver():
    """Context manager for driver with automatic cleanup"""
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-plugins")
    options.add_argument("--disable-images")
    options.add_argument("--memory-pressure-off")
    options.add_argument("--max_old_space_size=2048")
    driver = None
    try:
        driver = webdriver.Chrome(options=options)
        driver.implicitly_wait(5)
        yield driver
    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass

def extract_profile(driver, user_id, depth):
    """Optimized profile extraction with timeout"""
    url = f"https://scholar.google.com/citations?hl=en&user={user_id}"
    
    try:
        driver.set_page_load_timeout(15)
        driver.get(url)
        
        # Wait for key elements with timeout
        wait = WebDriverWait(driver, 10)
        wait.until(EC.presence_of_element_located((By.ID, "gsc_prf_in")))
        
        soup = BeautifulSoup(driver.page_source, "html.parser")

        name_tag = soup.select_one("#gsc_prf_in")
        position_tag = soup.select_one(".gsc_prf_il")
        email_tag = soup.select_one("#gsc_prf_ivh")
        interest_tags = soup.select("#gsc_prf_int a")
        metrics_rows = soup.select("#gsc_rsb_st tbody tr")

        name = name_tag.get_text(strip=True) if name_tag else "Unknown"
        position = position_tag.get_text(strip=True) if position_tag else "Unknown"
        email = email_tag.get_text(strip=True) if email_tag else "Unknown"
        interests_raw = ", ".join(tag.text for tag in interest_tags) if interest_tags else ""
        interest_phrases = normalize_interest_phrases(interests_raw)
        country = infer_country_from_email_field(email)
        institution = get_institution_from_email(email)

        citations_all = "0"
        h_index_all = "0"
        if metrics_rows and len(metrics_rows) >= 2:
            try:
                citations_all = metrics_rows[0].select("td")[1].text
                h_index_all = metrics_rows[1].select("td")[1].text
            except Exception:
                pass

        # Get coauthors (limited to prevent memory issues)
        coauthors = []
        coauthor_tags = soup.select(".gsc_rsb_aa .gsc_rsb_a_desc a")[:20]  # Limit coauthors
        for a_tag in coauthor_tags:
            href = a_tag.get("href", "")
            if "user=" in href:
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
            "topic_clusters": [],
            "search_depth": depth,
            "coauthors": coauthors,
        }
        return profile
        
    except Exception as e:
        logger.error(f"Error extracting profile for {user_id}: {e}")
        raise

@lru_cache(maxsize=1000)
def infer_country_from_email_field(email_field):
    """Cached country inference"""
    if not email_field:
        return "unknown"
    match = re.search(r"Verified email at ([^\s]+?)(?:\s*-\s*Homepage)?$", email_field)
    if not match:
        return "unknown"
    domain = match.group(1).lower().strip()
    ext = tldextract.extract(domain)
    suffix = ext.suffix.lower()
    if suffix in suffix_country_map:
        return suffix_country_map[suffix]
    else:
        last_part = suffix.split('.')[-1]
        return suffix_country_map.get(last_part, "unknown")

@lru_cache(maxsize=1000)
def get_institution_from_email(email_field):
    """Cached institution extraction"""
    if not email_field:
        return "Unknown"
    match = re.search(r"Verified email at ([^\s]+?)(?:\s*-\s*Homepage)?$", email_field)
    if not match:
        return "Unknown"
    domain = match.group(1).lower().strip()
    for known_domain in domain_to_institution:
        if domain.endswith(known_domain):
            return domain_to_institution[known_domain]
    return "Unknown"

def check_memory_usage():
    """Check if memory usage is getting too high"""
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    if memory_mb > MAX_MEMORY_MB:
        logger.warning(f"Memory usage high: {memory_mb:.1f}MB, forcing garbage collection")
        gc.collect()
        return True
    return False

def save_progress(profiles, _graph=None):
    """Optimized save with chunking for large datasets"""
    try:
        if len(profiles) > 10000:  # Use chunking for large datasets
            chunk_size = 5000
            temp_files = []
            for i in range(0, len(profiles), chunk_size):
                chunk = profiles[i:i+chunk_size]
                temp_file = f"temp_chunk_{i}.csv"
                pd.DataFrame(chunk).to_csv(temp_file, index=False)
                temp_files.append(temp_file)
            
            # Combine chunks
            combined_df = pd.concat([pd.read_csv(f) for f in temp_files], ignore_index=True)
            combined_df.to_csv(PROGRESS_CSV, index=False)
            
            # Clean up temp files
            for temp_file in temp_files:
                try:
                    os.remove(temp_file)
                except Exception:
                    pass
        else:
            df = pd.DataFrame(profiles)
            df.to_csv(PROGRESS_CSV, index=False)
        
        logger.info(f"💾 Progress saved: {len(profiles)} profiles")
        
        # Force garbage collection after save
        if check_memory_usage():
            time.sleep(1)  # Brief pause for memory cleanup
            
    except Exception as e:
        logger.error(f"Error saving progress: {e}")

# Cached fuzzy matching functions
fuzzy_cache = {}
FUZZY_CACHE_PATH = "fuzzy_match_cache.json"
ICLR_PARQUET_PATH = 'iclr_2020_2025_combined_data.parquet'
NEURIPS_PARQUET_PATH = 'neurips_2020_2024_combined_data.parquet'
FUZZY_RUN_INTERVAL = 3  # Run fuzzy matching more frequently

def load_cache(cache_path):
    """Load cache with error handling"""
    try:
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Error loading cache: {e}")
    return {}

def save_cache(cache, cache_path):
    """Save cache with error handling"""
    try:
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving cache: {e}")

def fuzzy_match_conference_participation(profile, conf_name, df, name_col='Author', inst_col='Institution', threshold=85):
    """Optimized fuzzy matching with caching"""
    cache_key = f"{profile.get('name', '')}_{conf_name}"
    if cache_key in fuzzy_cache:
        result = fuzzy_cache[cache_key]
        profile[f"Participated_in_{conf_name}"] = result['participated']
        profile[f"{conf_name}_Institution"] = result['institution']
        return

    authors = df[name_col].dropna().unique()
    authors_lower = [a.lower() for a in authors]

    profile_name = profile.get("name", "").lower()
    if not profile_name:
        profile[f"Participated_in_{conf_name}"] = False
        profile[f"{conf_name}_Institution"] = ""
        fuzzy_cache[cache_key] = {'participated': False, 'institution': ""}
        return

    try:
        match, score = process.extractOne(profile_name, authors_lower, scorer=fuzz.token_sort_ratio)
        if score >= threshold:
            match_idx = authors_lower.index(match)
            matched_author = authors[match_idx]
            matched_institution = df[df[name_col] == matched_author][inst_col].dropna().iloc[0] if inst_col in df.columns else ""
            profile[f"Participated_in_{conf_name}"] = True
            profile[f"{conf_name}_Institution"] = matched_institution
            fuzzy_cache[cache_key] = {'participated': True, 'institution': matched_institution}
            logger.info(f"✅ [{conf_name}] Match: {profile['name']} → {matched_author} @ {matched_institution}")
        else:
            profile[f"Participated_in_{conf_name}"] = False
            profile[f"{conf_name}_Institution"] = ""
            fuzzy_cache[cache_key] = {'participated': False, 'institution': ""}
    except Exception as e:
        logger.warning(f"Error in fuzzy matching for {profile_name}: {e}")
        profile[f"Participated_in_{conf_name}"] = False
        profile[f"{conf_name}_Institution"] = ""

def run_fuzzy_matching_single(profile):
    """Optimized single profile fuzzy matching"""
    if profile.get("Fuzzy_Matched"):
        return

    try:
        if os.path.exists(ICLR_PARQUET_PATH):
            iclr_df = pd.read_parquet(ICLR_PARQUET_PATH)
            fuzzy_match_conference_participation(profile, "ICLR", iclr_df)
        else:
            logger.warning("⚠️ ICLR Parquet not found")

        if os.path.exists(NEURIPS_PARQUET_PATH):
            neurips_df = pd.read_parquet(NEURIPS_PARQUET_PATH)
            fuzzy_match_conference_participation(profile, "NeurIPS", neurips_df)
        else:
            logger.warning("⚠️ NeurIPS Parquet not found")

        profile["Fuzzy_Matched"] = True
    except Exception as e:
        logger.error(f"Error in fuzzy matching: {e}")

# Queue management with file locking
import fcntl

def load_queue():
    """Thread-safe queue loading"""
    if not os.path.exists(QUEUE_FILE):
        return deque()
    
    try:
        with open(QUEUE_FILE, "r") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            lines = [line.strip() for line in f if line.strip()]
        
        queue = deque()
        for line in lines:
            try:
                data = json.loads(line)
                queue.append(tuple(data))
            except json.JSONDecodeError:
                queue.append((line, 0, None))
        return queue
    except Exception as e:
        logger.error(f"Error loading queue: {e}")
        return deque()

def save_queue(queue):
    """Thread-safe queue saving"""
    try:
        with open(QUEUE_FILE, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            for item in queue:
                f.write(json.dumps(item) + "\n")
    except Exception as e:
        logger.error(f"Error saving queue: {e}")

def enqueue_user(queue, user_id, depth, parent_id=None, visited_ids=None, prepend=False):
    """Optimized user enqueueing"""
    if visited_ids is None:
        visited_ids = set()
    if user_id in visited_ids:
        return
    if any(user_id == item[0] for item in queue):
        return
    new_item = (user_id, depth, parent_id)
    if prepend:
        queue.appendleft(new_item)
    else:
        queue.append(new_item)

def load_progress():
    """Memory-optimized progress loading"""
    all_profiles = []
    visited_depths = {}
    graph = nx.Graph()
    all_interest_phrases = defaultdict(list)

    # Load profiles CSV in chunks if large
    if os.path.exists(PROGRESS_CSV):
        try:
            file_size = os.path.getsize(PROGRESS_CSV)
            if file_size > 50 * 1024 * 1024:  # 50MB threshold
                logger.info("Large CSV detected, loading in chunks...")
                chunk_iter = pd.read_csv(PROGRESS_CSV, chunksize=1000)
                for chunk in chunk_iter:
                    all_profiles.extend(chunk.to_dict(orient="records"))
                    for _, p in chunk.iterrows():
                        visited_depths[p["user_id"]] = p.get("search_depth", 0)
            else:
                profiles_df = pd.read_csv(PROGRESS_CSV)
                all_profiles = profiles_df.to_dict(orient="records")
                for p in all_profiles:
                    visited_depths[p["user_id"]] = p.get("search_depth", 0)
            
            logger.info(f"✅ Loaded {len(all_profiles)} profiles from CSV")
        except Exception as e:
            logger.error(f"⚠️ Error loading profiles CSV: {e}")

    # Load GraphML safely
    if os.path.exists(GRAPH_GRAPHML):
        try:
            graph = nx.read_graphml(GRAPH_GRAPHML)
        except Exception as e:
            logger.warning(f"⚠️ Error loading graphml file: {e}")
            graph = nx.Graph()

    # Rebuild interest phrases more efficiently
    for p in all_profiles:
        phrases = p.get("interest_phrases", [])
        if isinstance(phrases, str):
            try:
                phrases = eval(phrases)
            except:
                phrases = []
        for phrase in phrases:
            if phrase:
                all_interest_phrases[phrase].append(p["user_id"])

    return all_profiles, visited_depths, graph, all_interest_phrases

def sanitize_profile_data(profile):
    """Optimized profile data sanitization"""
    cleaned = {}
    for k, v in profile.items():
        if isinstance(v, list):
            cleaned[k] = ", ".join(str(x) for x in v[:10])  # Limit list size
        elif isinstance(v, dict):
            cleaned[k] = json.dumps(v)[:500]  # Limit dict size
        elif isinstance(v, type):
            cleaned[k] = str(v)
        elif v is None:
            cleaned[k] = "Unknown"
        else:
            cleaned[k] = str(v)[:1000]  # Limit string length
    return cleaned

def get_coauthors_from_profile(driver, user_id):
    """Optimized coauthor extraction"""
    url = f"https://scholar.google.com/citations?hl=en&user={user_id}"
    try:
        driver.set_page_load_timeout(10)
        driver.get(url)
        
        wait = WebDriverWait(driver, 8)
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "gsc_rsb_aa")))
        
        soup = BeautifulSoup(driver.page_source, "html.parser")
        coauthors = []
        
        # Limit coauthors to prevent memory issues
        coauthor_tags = soup.select(".gsc_rsb_aa .gsc_rsb_a_desc a")[:15]
        for a_tag in coauthor_tags:
            href = a_tag.get("href", "")
            if "user=" in href:
                co_id = href.split("user=")[1].split("&")[0]
                coauthors.append(co_id)
        
        logger.info(f"✅ Found {len(coauthors)} co-authors for {user_id}")
        return coauthors
        
    except Exception as e:
        logger.warning(f"Failed to fetch co-authors for {user_id}: {e}")
        return []

def crawl_bfs_resume(driver, queue, all_profiles, visited_depths, force=False):
    """Optimized BFS crawling with memory management"""
    start_time = time.time()
    total_scraped = 0
    INSERT_FILE = "queue_insert.jsonl"
    INSERT_CHECK_INTERVAL = 30
    last_insert_check = time.time()
    last_memory_check = time.time()
    MEMORY_CHECK_INTERVAL = 60  # Check memory every minute

    # Memory-efficient visited_ids creation
    visited_ids = set()
    for i, p in enumerate(all_profiles):
        if isinstance(p, dict):
            visited_ids.add(p.get("user_id"))
        elif isinstance(p, tuple):
            visited_ids.add(p[0])
        
        # Periodic memory check during initialization
        if i % 1000 == 0:
            check_memory_usage()

    # Force re-processing logic
    if force:
        queued_user_ids = set(item[0] for item in queue)
        for uid in queued_user_ids:
            if uid in visited_depths:
                logger.info(f"⚠️ Forcing re-process of user {uid}")
                visited_depths.pop(uid)

    profile_batch = []
    
    while queue:
        current_time = time.time()
        
        # Memory management
        if current_time - last_memory_check > MEMORY_CHECK_INTERVAL:
            if check_memory_usage():
                # Save current batch if memory is high
                if profile_batch:
                    all_profiles.extend(profile_batch)
                    save_progress(all_profiles)
                    profile_batch = []
                
                # Clear caches
                if len(author_info_cache) > 500:
                    author_info_cache.clear()
                if len(fuzzy_cache) > 1000:
                    fuzzy_cache.clear()
                
                gc.collect()
            last_memory_check = current_time

        # Check for new insertions
        if current_time - last_insert_check > INSERT_CHECK_INTERVAL:
            if os.path.exists(INSERT_FILE):
                try:
                    with open(INSERT_FILE, "r") as f:
                        new_lines = [line.strip() for line in f if line.strip()]
                    os.remove(INSERT_FILE)

                    new_items = []
                    for line in new_lines:
                        try:
                            user_id, depth, parent = json.loads(line)
                            if user_id not in visited_depths and not any(user_id == q[0] for q in queue):
                                new_items.append((user_id, depth, parent))
                        except Exception as e:
                            logger.warning(f"⚠️ Invalid line in insert file: {line} — {e}")
                    
                    if new_items:
                        logger.info(f"📥 Inserting {len(new_items)} new users into the queue")
                        for item in reversed(new_items):
                            queue.appendleft(item)
                except Exception as e:
                    logger.error(f"❌ Error reading from insert file: {e}")
            last_insert_check = current_time

        # Time limit check
        elapsed = current_time - start_time
        if elapsed > MAX_CRAWL_SECONDS:
            logger.info(f"🛑 Max crawl time {MAX_CRAWL_SECONDS}s reached, ending crawl.")
            break

        user_id, depth, parent_id = queue.popleft()
        depth = int(depth)

        # Skip if already visited
        if not force and user_id in visited_depths:
            logger.info(f"Skipping already visited user {user_id}")
            continue

        logger.info(f"🔎 Crawling {user_id} at depth {depth}")
        
        try:
            profile = extract_profile(driver, user_id, depth)
            visited_depths[user_id] = depth

            # Run fuzzy matching
            run_fuzzy_matching_single(profile)

            # Get Wikipedia info with error handling
            try:
                wiki_info = get_author_wikipedia_info(profile.get("name", ""))
                profile.update({
                    "wiki_birth_name": wiki_info.get("birth_name", ""),
                    "wiki_name": wiki_info.get("name", ""),
                    "wiki_birth_date": wiki_info.get("birth_date", ""),
                    "wiki_birth_place": wiki_info.get("birth_place", ""),
                    "wiki_death_date": wiki_info.get("death_date", ""),
                    "wiki_death_place": wiki_info.get("death_place", ""),
                    "wiki_fields": wiki_info.get("fields", ""),
                    "wiki_work_institution": wiki_info.get("work_institution", ""),
                    "wiki_alma_mater": wiki_info.get("alma_mater", ""),
                    "wiki_notable_students": wiki_info.get("notable_students", ""),
                    "wiki_thesis_title": wiki_info.get("thesis_title", ""),
                    "wiki_thesis_year": wiki_info.get("thesis_year", ""),
                    "wiki_thesis_url": wiki_info.get("thesis_url", ""),
                    "wiki_known_for": wiki_info.get("known_for", ""),
                    "wiki_awards": wiki_info.get("awards", ""),
                    "wiki_deceased": wiki_info.get("deceased", False),
                    "wiki_wiki_summary": wiki_info.get("wiki_summary", ""),
                    "wiki_is_researcher_ml": wiki_info.get("is_researcher_ml", False),
                    "wiki_matched_title": wiki_info.get("matched_title", None)
                })
            except Exception as e:
                logger.warning(f"Error getting Wikipedia info for {profile.get('name', '')}: {e}")
                # Set default values
                wiki_fields = ["wiki_birth_name", "wiki_name", "wiki_birth_date", "wiki_birth_place", 
                              "wiki_death_date", "wiki_death_place", "wiki_fields", "wiki_work_institution",
                              "wiki_alma_mater", "wiki_notable_students", "wiki_thesis_title", "wiki_thesis_year",
                              "wiki_thesis_url", "wiki_known_for", "wiki_awards", "wiki_wiki_summary", "wiki_matched_title"]
                for field in wiki_fields:
                    profile[field] = ""
                profile["wiki_deceased"] = False
                profile["wiki_is_researcher_ml"] = False

            # Add to batch
            profile_batch.append(profile)
            total_scraped += 1

            logger.info(f"✅ {profile['name']} | h-index: {profile['h_index_all']} | {profile['country']} | depth {depth}")

            # Enqueue co-authors with limits
            coauthors = profile.get("coauthors", [])[:15]  # Limit coauthors
            for co_id in coauthors:
                if co_id not in visited_depths:
                    enqueue_user(queue, co_id, depth + 1, user_id)

            # Process batch when it reaches size limit
            if len(profile_batch) >= BATCH_SIZE:
                all_profiles.extend(profile_batch)
                save_progress(all_profiles)
                save_queue(queue)
                profile_batch = []
                logger.info(f"💾 Batch processed: {total_scraped} profiles total")

        except Exception as e:
            logger.error(f"❌ Error scraping {user_id}: {e}")
            continue

    # Process remaining batch
    if profile_batch:
        all_profiles.extend(profile_batch)
        save_progress(all_profiles)

    logger.info(f"✅ BFS crawl finished with {total_scraped} new profiles.")
    save_queue(queue)
    return all_profiles

# Optimized clustering with memory management
def perform_clustering(all_profiles, model):
    """Memory-efficient clustering"""
    logger.info("📊 Starting phrase clustering...")
    
    # Collect phrases more efficiently
    all_interest_phrases = []
    for profile in all_profiles:
        phrases = profile.get("interest_phrases", [])
        if isinstance(phrases, str):
            try:
                phrases = eval(phrases)
            except:
                phrases = []
        all_interest_phrases.extend(phrases)
    
    unique_phrases = list(set(all_interest_phrases))
    logger.info(f"📊 Clustering {len(unique_phrases)} unique phrases...")

    if len(unique_phrases) >= 2:
        try:
            # Process in batches to manage memory
            batch_size = 1000
            all_embeddings = []
            
            for i in range(0, len(unique_phrases), batch_size):
                batch = unique_phrases[i:i+batch_size]
                batch_embeddings = model.encode(batch)
                all_embeddings.append(batch_embeddings)
                
                # Memory check
                if check_memory_usage():
                    gc.collect()
            
            # Combine embeddings
            embeddings = np.vstack(all_embeddings)
            
            # Perform clustering
            n_clusters = min(NUM_PHRASE_CLUSTERS, len(unique_phrases))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(embeddings)
            phrase_to_cluster = {phrase: int(label) for phrase, label in zip(unique_phrases, kmeans.labels_)}

            # Update profiles with clusters
            for profile in all_profiles:
                phrases = profile.get("interest_phrases", [])
                if isinstance(phrases, str):
                    try:
                        phrases = eval(phrases)
                    except:
                        phrases = []
                
                clusters = sorted(set(phrase_to_cluster.get(ph) for ph in phrases if ph in phrase_to_cluster and phrase_to_cluster.get(ph) is not None))
                profile["topic_clusters"] = clusters
            
            logger.info("✅ Clustering completed successfully")
            
        except Exception as e:
            logger.error(f"Error in clustering: {e}")
            # Set empty clusters as fallback
            for profile in all_profiles:
                profile["topic_clusters"] = []
    
    return all_profiles

# === MAIN EXECUTION ===
if __name__ == "__main__":
    log_memory_usage("startup")
    
    # Initialize model with memory optimization
    logger.info("Loading sentence transformer model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Load fuzzy cache
    fuzzy_cache = load_cache(FUZZY_CACHE_PATH)
    
    # Initialize progress tracking
    ensure_progress_csv(PROGRESS_CSV)
    all_profiles, visited_depths, graph, all_interest_phrases = load_progress()
    
    log_memory_usage("after loading")
    
    with get_driver() as driver:
        progress_bar = tqdm(total=len(all_profiles), desc="Crawling profiles", dynamic_ncols=True)
        
        try:
            # Initialize queue
            queue = deque()
            
            if all_profiles:
                # Get coauthors from recent profiles
                last_users = [p["user_id"] for p in all_profiles[-5:]]  # Reduced from 10 to 5
                max_depth = max(visited_depths.get(uid, 0) for uid in last_users)
                logger.info(f"🔍 Fetching co-authors of last 5 users: {last_users}")
            
                coauthors = set()
                for user_id in last_users:
                    try:
                        new_coauthors = get_coauthors_from_profile(driver, user_id)
                        for co in new_coauthors:
                            if co not in visited_depths:
                                coauthors.add(co)
                    except Exception as e:
                        logger.error(f"❌ Failed to fetch co-authors for {user_id}: {e}")
            
                if coauthors:
                    for co in list(coauthors)[:20]:  # Limit initial queue size
                        queue.append((co, max_depth + 1, None))
                    logger.info(f"✅ Queue initialized with {len(queue)} new co-authors at depth {max_depth + 1}.")
                else:
                    logger.info(f"⚠️ No new co-authors found, starting from seed user {SEED_USER_ID}")
                    queue.append((SEED_USER_ID, 0, None))
            else:
                logger.info(f"🔄 Starting fresh from seed {SEED_USER_ID}")
                queue.append((SEED_USER_ID, 0, None))

            # Main crawling
            all_profiles = crawl_bfs_resume(driver, queue, all_profiles, visited_depths, force=True)
            
            log_memory_usage("after crawling")
            
            # Perform clustering
            all_profiles = perform_clustering(all_profiles, model)
            
            # Final save
            save_progress(all_profiles)
            save_cache(fuzzy_cache, FUZZY_CACHE_PATH)
            
            logger.info("✅ All processing complete!")
            log_memory_usage("final")

        except KeyboardInterrupt:
            logger.info("🛑 Interrupted by user, saving progress...")
            save_progress(all_profiles)
            save_cache(fuzzy_cache, FUZZY_CACHE_PATH)
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
            save_progress(all_profiles)
            save_cache(fuzzy_cache, FUZZY_CACHE_PATH)
        finally:
            progress_bar.close()
            session_manager.close()
            
            # Final cleanup
            gc.collect()
            log_memory_usage("cleanup complete")
