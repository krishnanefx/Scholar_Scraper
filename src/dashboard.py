import streamlit as st
import os
import re
import json
import pandas as pd
import numpy as np
from collections import Counter
from fuzzywuzzy import process
import ast
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from datetime import datetime
import plotly.express as px

# === Coauthor Name Mapping Utility ===
def build_userid_to_name_map(df):
    """Build a mapping from user_id to name for all researchers."""
    if "user_id" in df.columns and "name" in df.columns:
        return dict(zip(df["user_id"].astype(str), df["name"].astype(str)))
    return {}

def coauthor_names_from_str(coauthors_str, userid_to_name):
    """Convert coauthor user_ids in a string/list to names, fallback to user_id if name not found."""
    import ast
    if not coauthors_str or pd.isna(coauthors_str):
        return []
    try:
        if isinstance(coauthors_str, str) and coauthors_str.startswith("["):
            coauthor_list = ast.literal_eval(coauthors_str)
        elif isinstance(coauthors_str, list):
            coauthor_list = coauthors_str
        else:
            return []
        names = []
        for coauthor in coauthor_list:
            if isinstance(coauthor, dict):
                uid = str(coauthor.get("user_id", ""))
                name = coauthor.get("name") or userid_to_name.get(uid, uid)
                if name:
                    names.append(name)
            else:
                uid = str(coauthor)
                name = userid_to_name.get(uid, uid)
                if name:
                    names.append(name)
        return names
    except Exception:
        return []

# === Page Configuration ===
st.set_page_config(
    page_title="AI Researcher Database",
    page_icon="📚",
    layout="wide",
)

# === Constants ===
def get_profiles_file_path():
    """Get the correct path to the profiles file, works both locally and when deployed"""
    # Try multiple possible paths
    possible_paths = [
        "../data/processed/scholar_profiles.csv",  # Local development
        "data/processed/scholar_profiles.csv",     # Deployed from repo root
        "./data/processed/scholar_profiles.csv",   # Current directory
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "scholar_profiles.csv"),  # Absolute from script location
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # If no file found, return the most likely path for error messages
    return "data/processed/scholar_profiles.csv"

PROFILES_FILE = get_profiles_file_path()

# === Cache for DataFrame ===
_PROFILES_DF_CACHE = None

# Clear cache to force reload with country standardization
def clear_profiles_cache():
    """Clear the cached profiles DataFrame to force reload"""
    global _PROFILES_DF_CACHE
    _PROFILES_DF_CACHE = None

def get_profiles_df():
    """Load profiles CSV once and cache result to avoid repeated disk I/O."""
    global _PROFILES_DF_CACHE
    if _PROFILES_DF_CACHE is None:
        if os.path.exists(PROFILES_FILE):
            try:
                df = pd.read_csv(PROFILES_FILE, engine='python', on_bad_lines='skip')
                df.replace(["NaN", "nan", ""], np.nan, inplace=True)
                df.columns = [c.strip() for c in df.columns]
                
                # Standardize country names
                if 'country' in df.columns:
                    df['country'] = df['country'].apply(standardize_country_name)
                
                _PROFILES_DF_CACHE = df
            except Exception as e:
                st.error(f"Error loading profiles: {e}")
                _PROFILES_DF_CACHE = pd.DataFrame()
        else:
            _PROFILES_DF_CACHE = pd.DataFrame()
    return _PROFILES_DF_CACHE

# === Country Standardization ===
COUNTRY_MAPPING = {
    'at': 'Austria',
    'au': 'Australia',
    'ca': 'Canada',
    'cn': 'China',
    'de': 'Germany',
    'fr': 'France',
    'jp': 'Japan',
    'kr': 'South Korea',
    'nz': 'New Zealand',
    'sg': 'Singapore',
    'uk': 'United Kingdom',
    'us': 'United States',
    # Add any existing full names that should remain unchanged
    'austria': 'Austria',
    'australia': 'Australia',
    'canada': 'Canada',
    'china': 'China',
    'germany': 'Germany',
    'france': 'France',
    'japan': 'Japan',
    'south korea': 'South Korea',
    'new zealand': 'New Zealand',
    'singapore': 'Singapore',
    'united kingdom': 'United Kingdom',
    'united states': 'United States',
}

def standardize_country_name(country):
    """Convert country codes to full country names"""
    if pd.isna(country):
        return country
    country_lower = str(country).lower().strip()
    return COUNTRY_MAPPING.get(country_lower, country)

# === Award and Fellowship Definitions ===
PRESTIGIOUS_AWARDS_MATCH = [
    "nobel prize", "turing award", "fields medal", "rumelhart prize",
    "princess of asturias award", "acm a.m. turing award", "ieee john von neumann medal",
    "gödel prize", "acm prize in computing", "knuth prize", "acm grace murray hopper award",
    "c&c prize", "dijkstra prize", "nsf career award",
    "ieee computer society seymour cray computer engineering award",
    "siggraph computer graphics achievement award", "vinfuture prize",
    "william bowie medal", "wolf prize in physics", "wollaston medal"
]

PRESTIGIOUS_AWARDS_DISPLAY = {
    "nobel prize": "🏅 Nobel Prize",
    "turing award": "🏅 Turing Award",
    "fields medal": "🏅 Fields Medal",
    "rumelhart prize": "🏅 Rumelhart Prize",
    "princess of asturias award": "🏅 Princess Of Asturias Award",
    "acm a.m. turing award": "🏅 ACM A.M. Turing Award",
    "ieee john von neumann medal": "🏅 IEEE John von Neumann Medal",
    "gödel prize": "🏅 Gödel Prize",
    "acm prize in computing": "🏅 ACM Prize in Computing",
    "knuth prize": "🏅 Knuth Prize",
    "acm grace murray hopper award": "🏅 ACM Grace Murray Hopper Award",
    "c&c prize": "🏅 C&C Prize",
    "dijkstra prize": "🏅 Dijkstra Prize",
    "nsf career award": "🏅 NSF CAREER Award",
    "ieee computer society seymour cray computer engineering award": "🏅 IEEE Seymour Cray Award",
    "siggraph computer graphics achievement award": "🏅 SIGGRAPH Achievement Award",
    "vinfuture prize": "🏅 VinFuture Prize",
    "william bowie medal": "🏅 William Bowie Medal",
    "wolf prize in physics": "🏅 Wolf Prize in Physics",
    "wollaston medal": "🏅 Wollaston Medal"
}

AWARD_COLORS = {
    "nobel prize": "#b8860b", "turing award": "#1e90ff", "fields medal": "#32cd32",
    "rumelhart prize": "#ff69b4", "princess of asturias award": "#8a2be2",
    "acm a.m. turing award": "#1e90ff", "ieee john von neumann medal": "#4682b4",
    "gödel prize": "#6a5acd", "acm prize in computing": "#20b2aa",
    "knuth prize": "#ff8c00", "acm grace murray hopper award": "#da70d6",
    "c&c prize": "#ff6347", "dijkstra prize": "#00ced1",
    "nsf career award": "#9acd32",
    "ieee computer society seymour cray computer engineering award": "#ff4500",
    "siggraph computer graphics achievement award": "#ff1493",
    "vinfuture prize": "#9370db", "william bowie medal": "#8b4513",
    "wolf prize in physics": "#b8860b", "wollaston medal": "#daa520"
}

PRESTIGIOUS_FELLOWSHIPS_MATCH = [
    "guggenheim fellowship", "macarthur fellowship", "sloan research fellowship",
    "packard fellowship", "nsf fellowship", "doe computational science graduate fellowship",
    "hertz fellowship", "stanford graduate fellowship", "google phd fellowship",
    "facebook fellowship", "microsoft research phd fellowship", "nvidia graduate fellowship",
    "apple scholars in aiml", "simons foundation fellowship", "simons investigator",
    "moore foundation fellowship", "chan zuckerberg biohub investigator",
    "chan zuckerberg initiative", "packard fellowships for science and engineering",
    "searle scholars program", "beckman young investigator",
    "arnold and mabel beckman foundation", "rita allen foundation scholar",
    "pew biomedical scholar", "james s. mcdonnell foundation", "templeton foundation",
    "royal society fellowship", "royal society research fellow",
    "leverhulme trust fellowship", "wellcome trust", "european research council",
    "erc starting grant", "erc consolidator grant", "erc advanced grant",
    "marie curie fellowship", "humboldt fellowship", "fulbright fellowship",
    "rhodes scholarship", "marshall scholarship", "churchill scholarship",
    "gates cambridge scholarship", "knight-hennessy scholars", "schwarzman scholars",
    "ieee fellow", "aaai fellow", "acm fellow", "wwrf fellow"
]

PRESTIGIOUS_FELLOWSHIPS_DISPLAY = {
    "guggenheim fellowship": "🎓 Guggenheim Fellowship",
    "macarthur fellowship": "🎓 MacArthur Fellowship",
    "sloan research fellowship": "🎓 Sloan Research Fellowship",
    "packard fellowship": "🎓 Packard Fellowship",
    "nsf fellowship": "🎓 NSF Fellowship",
    "doe computational science graduate fellowship": "🎓 DOE CSGF",
    "hertz fellowship": "🎓 Hertz Fellowship",
    "stanford graduate fellowship": "🎓 Stanford Graduate Fellowship",
    "google phd fellowship": "🎓 Google PhD Fellowship",
    "facebook fellowship": "🎓 Facebook Fellowship",
    "microsoft research phd fellowship": "🎓 Microsoft Research PhD Fellowship",
    "nvidia graduate fellowship": "🎓 NVIDIA Graduate Fellowship",
    "apple scholars in aiml": "🎓 Apple Scholars in AIML",
    "simons foundation fellowship": "🎓 Simons Foundation Fellowship",
    "simons investigator": "🎓 Simons Investigator",
    "moore foundation fellowship": "🎓 Moore Foundation Fellowship",
    "chan zuckerberg biohub investigator": "🎓 CZ Biohub Investigator",
    "chan zuckerberg initiative": "🎓 Chan Zuckerberg Initiative",
    "packard fellowships for science and engineering": "🎓 Packard Fellowship",
    "searle scholars program": "🎓 Searle Scholar",
    "beckman young investigator": "🎓 Beckman Young Investigator",
    "arnold and mabel beckman foundation": "🎓 Beckman Foundation",
    "rita allen foundation scholar": "🎓 Rita Allen Scholar",
    "pew biomedical scholar": "🎓 Pew Biomedical Scholar",
    "james s. mcdonnell foundation": "🎓 McDonnell Foundation",
    "templeton foundation": "🎓 Templeton Foundation",
    "royal society fellowship": "🎓 Royal Society Fellowship",
    "royal society research fellow": "🎓 Royal Society Research Fellow",
    "leverhulme trust fellowship": "🎓 Leverhulme Trust Fellowship",
    "wellcome trust fellowship": "🎓 Wellcome Trust Fellowship",
    "european research council": "🎓 European Research Council",
    "erc starting grant": "🎓 ERC Starting Grant",
    "erc consolidator grant": "🎓 ERC Consolidator Grant",
    "erc advanced grant": "🎓 ERC Advanced Grant",
    "marie curie fellowship": "🎓 Marie Curie Fellowship",
    "humboldt fellowship": "🎓 Humboldt Fellowship",
    "fulbright fellowship": "🎓 Fulbright Fellowship",
    "rhodes scholarship": "🎓 Rhodes Scholarship",
    "marshall scholarship": "🎓 Marshall Scholarship",
    "churchill scholarship": "🎓 Churchill Scholarship",
    "gates cambridge scholarship": "🎓 Gates Cambridge Scholarship",
    "knight-hennessy scholars": "🎓 Knight-Hennessy Scholars",
    "schwarzman scholars": "🎓 Schwarzman Scholars",
    "ieee fellow": "🎓 IEEE Fellow",
    "aaai fellow": "🎓 AAAI Fellow",
    "acm fellow": "🎓 ACM Fellow",
    "wwrf fellow": "🎓 WWRF Fellow"
}

FELLOWSHIP_COLORS = {
    "guggenheim fellowship": "#8b4513", "macarthur fellowship": "#800080",
    "sloan research fellowship": "#4169e1", "packard fellowship": "#228b22",
    "nsf fellowship": "#ff6347", "doe computational science graduate fellowship": "#2e8b57",
    "hertz fellowship": "#dc143c", "stanford graduate fellowship": "#8b0000",
    "google phd fellowship": "#4285f4", "facebook fellowship": "#1877f2",
    "microsoft research phd fellowship": "#0078d4", "nvidia graduate fellowship": "#76b900",
    "apple scholars in aiml": "#007aff", "simons foundation fellowship": "#ff8c00",
    "simons investigator": "#ffa500", "moore foundation fellowship": "#32cd32",
    "chan zuckerberg biohub investigator": "#1e90ff", "chan zuckerberg initiative": "#0080ff",
    "packard fellowships for science and engineering": "#228b22",
    "searle scholars program": "#4682b4", "beckman young investigator": "#6a5acd",
    "arnold and mabel beckman foundation": "#9370db", "rita allen foundation scholar": "#da70d6",
    "pew biomedical scholar": "#20b2aa", "james s. mcdonnell foundation": "#b8860b",
    "templeton foundation": "#ff69b4", "royal society fellowship": "#8a2be2",
    "royal society research fellow": "#9932cc", "leverhulme trust fellowship": "#4b0082",
    "wellcome trust fellowship": "#00ced1", "european research council": "#0000cd",
    "erc starting grant": "#0000ff", "erc consolidator grant": "#4169e1",
    "erc advanced grant": "#191970", "marie curie fellowship": "#ff1493",
    "humboldt fellowship": "#ffd700", "fulbright fellowship": "#ff4500",
    "rhodes scholarship": "#00008b", "marshall scholarship": "#800000",
    "churchill scholarship": "#2f4f4f", "gates cambridge scholarship": "#008b8b",
    "knight-hennessy scholars": "#8b0000", "schwarzman scholars": "#000000",
    "ieee fellow": "#004c99", "aaai fellow": "#ff6600",
    "acm fellow": "#1e90ff", "wwrf fellow": "#800080"
}

# === Utility Functions ===
def extract_prestigious_awards(award_str):
    """Extract prestigious awards from a string"""
    if not award_str or not isinstance(award_str, str):
        return []
    chunks = [chunk.strip().lower() for chunk in re.split(r";|,| and |\n", award_str) if chunk.strip()]
    found = []
    for award in PRESTIGIOUS_AWARDS_MATCH:
        for chunk in chunks:
            if award in chunk:
                found.append(award)
                break
    return found

def extract_prestigious_fellowships(fellowship_str):
    """Extract prestigious fellowships from a string"""
    if not fellowship_str or not isinstance(fellowship_str, str):
        return []
    chunks = [chunk.strip().lower() for chunk in re.split(r";|,| and |\n", fellowship_str) if chunk.strip()]
    found = []
    for fellowship in PRESTIGIOUS_FELLOWSHIPS_MATCH:
        for chunk in chunks:
            if fellowship in chunk:
                found.append(fellowship)
                break
    return found

def render_colored_tags(tag_keys, colors=None):
    """Render colored tags for awards and fellowships"""
    colors = colors or {}
    tag_html = []
    for key in tag_keys:
        if key in PRESTIGIOUS_AWARDS_DISPLAY:
            display_name = PRESTIGIOUS_AWARDS_DISPLAY[key]
            color = AWARD_COLORS.get(key, "#6c757d")
        elif key in PRESTIGIOUS_FELLOWSHIPS_DISPLAY:
            display_name = PRESTIGIOUS_FELLOWSHIPS_DISPLAY[key]
            color = FELLOWSHIP_COLORS.get(key, "#6c757d")
        else:
            display_name = key.title()
            color = colors.get(key, "#6c757d")
        
        safe_tag = display_name.replace('"', '&quot;')
        tag_html.append(
            f'<span style="background-color:{color}; color:white; padding:3px 8px; border-radius:10px; margin-right:5px; font-size:0.85em;">{safe_tag}</span>'
        )
    return " ".join(tag_html)

def create_sample_profiles_file():
    """Create a sample profiles file for demonstration purposes"""
    try:
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(PROFILES_FILE), exist_ok=True)
        
        # Sample data
        sample_data = [
            {
                "user_id": "sample1",
                "name": "Dr. Jane Smith",
                "institution": "Stanford University",
                "country": "United States",
                "h_index_all": "45",
                "citations_all": "12500",
                "research_interests": "machine learning, computer vision",
                "wiki_matched_title": "Jane Smith (computer scientist)",
                "NeurIPS_Institution": "Stanford University",
                "ICLR_Institution": "",
                "coauthors": "[]"
            },
            {
                "user_id": "sample2", 
                "name": "Prof. John Doe",
                "institution": "MIT",
                "country": "United States",
                "h_index_all": "67",
                "citations_all": "23400",
                "research_interests": "artificial intelligence, robotics",
                "wiki_matched_title": "",
                "NeurIPS_Institution": "MIT",
                "ICLR_Institution": "MIT",
                "coauthors": "[\"sample1\"]"
            },
            {
                "user_id": "sample3",
                "name": "Dr. Li Wei",
                "institution": "National University of Singapore",
                "country": "Singapore", 
                "h_index_all": "32",
                "citations_all": "8900",
                "research_interests": "natural language processing, deep learning",
                "wiki_matched_title": "",
                "NeurIPS_Institution": "",
                "ICLR_Institution": "National University of Singapore",
                "coauthors": "[\"sample1\", \"sample2\"]"
            }
        ]
        
        # Create DataFrame and save
        df = pd.DataFrame(sample_data)
        df.to_csv(PROFILES_FILE, index=False)
        
        st.success(f"✅ Sample data created at: {PROFILES_FILE}")
        st.info("🔄 Please refresh the page to see the sample data.")
        
    except Exception as e:
        st.error(f"❌ Error creating sample file: {e}")

def get_display_columns(df):
    """Get columns for display, excluding specified columns"""
    excluded_cols = {"interest_phrases", "wiki_birth_name", "wiki_matched_title", 
                     "prestigious_awards_count", "prestigious_fellowships_count"}
    return [col for col in df.columns if col not in excluded_cols]

def fuzzy_author_search(df, query, score_cutoff=70):
    """Fuzzy search for authors by name"""
    if df.empty or not query.strip():
        return pd.DataFrame()
    names = df["name"].dropna().unique()
    results = process.extract(query, names, limit=100)
    filtered = [r for r in results if r[1] >= score_cutoff]
    matched_names = [r[0] for r in filtered]
    if not matched_names:
        return pd.DataFrame()
    return df[df["name"].isin(matched_names)].copy()

def fuzzy_institution_search(df, query, score_cutoff=70):
    """Fuzzy search for authors by institution"""
    if df.empty or not query.strip() or "institution" not in df.columns:
        return pd.DataFrame()
    institutions = df["institution"].dropna().unique()
    results = process.extract(query, institutions, limit=100)
    filtered = [r for r in results if r[1] >= score_cutoff]
    matched_insts = [r[0] for r in filtered]
    if not matched_insts:
        return pd.DataFrame()
    return df[df["institution"].isin(matched_insts)].copy()

# === Main App ===
def main():
    # Clear cache on app start to ensure country standardization takes effect
    clear_profiles_cache()
    
    st.title("📚 AI Researcher Database")
    st.markdown("### Comprehensive Academic Profile Analytics Dashboard")
    
    # === Main Content Tabs ===
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "🔍 Search", "📈 Analytics", "🌐 Network", "📥 Downloads"
    ])
    
    with tab1:
        show_overview()
    
    with tab2:
        show_search()
    
    with tab3:
        show_analytics()
    
    with tab4:
        show_network()
    
    with tab5:
        show_downloads()

def show_overview():
    """Display overview statistics and data"""
    st.header("📊 Enhanced Profile Stats")
    
    if not os.path.exists(PROFILES_FILE):
        st.error("📂 No profiles file found yet.")
        st.info("""
        **Possible solutions:**
        1. **If running locally:** Start crawling to generate the file
        2. **If deployed on Streamlit Cloud:**
           - Make sure the `data/processed/scholar_profiles.csv` file exists in your repository
           - The file should be committed and pushed to GitHub
           - Check that the file is not in `.gitignore`
        3. **Expected file path:** `{}`
        
        **For Streamlit Cloud deployment:**
        - Your repository should have the structure: `data/processed/scholar_profiles.csv`
        - You can upload your generated CSV file to this path in your GitHub repository
        """.format(PROFILES_FILE))
        
        # Suggest creating a sample file for demonstration
        if st.button("📝 Create Sample Data for Demo"):
            create_sample_profiles_file()
        return
    
    df = get_profiles_df()
    if df.empty:
        st.error("📂 Profiles file is empty or couldn't be loaded.")
        return
    
    # === Main Metrics ===
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("👤 Total Profiles", f"{len(df):,}")
    
    with col2:
        if "h_index_all" in df.columns:
            try:
                avg_h = round(df["h_index_all"].dropna().astype(float).mean(), 1)
                st.metric("📈 Avg H-Index", avg_h)
            except:
                st.metric("📈 Avg H-Index", "N/A")
    
    with col3:
        if "wiki_matched_title" in df.columns:
            wiki_count = df["wiki_matched_title"].notna().sum()
            st.metric("🌐 Wikipedia Matches", f"{wiki_count:,}")

    
    # === Conference Participation ===
    st.subheader("🎯 Conference Participation")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if "NeurIPS_Institution" in df.columns:
            total_neurips = 35609
            neurips_count = df["NeurIPS_Institution"].notna().sum()
            neurips_percent = neurips_count / total_neurips * 100
            
            help_text = f"{neurips_percent:.1f}% of NeurIPS participants found"
            st.metric(
                "🧠 NeurIPS Presenters", 
                f"{neurips_count:,} / {total_neurips:,}",
                help=help_text
            )
            
            # Add wiki match count for NeurIPS presenters
            if "wiki_matched_title" in df.columns:
                neurips_wiki_count = len(df[df["NeurIPS_Institution"].notna() & df["wiki_matched_title"].notna()])
                st.caption(f"📖 {neurips_wiki_count:,} with Wikipedia pages")
    
    with col2:
        if "ICLR_Institution" in df.columns:
            total_iclr = 27907
            iclr_count = df["ICLR_Institution"].notna().sum()
            iclr_percent = iclr_count / total_iclr * 100
            
            help_text = f"{iclr_percent:.1f}% of ICLR participants found"
            st.metric(
                "📖 ICLR Presenters", 
                f"{iclr_count:,} / {total_iclr:,}",
                help=help_text
            )
            
            # Add wiki match count for ICLR presenters
            if "wiki_matched_title" in df.columns:
                iclr_wiki_count = len(df[df["ICLR_Institution"].notna() & df["wiki_matched_title"].notna()])
                st.caption(f"📖 {iclr_wiki_count:,} with Wikipedia pages")
    
    with col3:
        if "NeurIPS_Institution" in df.columns and "ICLR_Institution" in df.columns:
            both_count = len(df[df["NeurIPS_Institution"].notna() & df["ICLR_Institution"].notna()])
            total_both = 13182
            st.metric(
                "🔗 Both Conferences", 
                f"{both_count:,} / {total_both:,}",
                help=f"{both_count / total_both * 100:.1f}% of dual participants found"
            )
            
            # Add wiki match count for both conferences
            if "wiki_matched_title" in df.columns:
                both_wiki_count = len(df[df["NeurIPS_Institution"].notna() & df["ICLR_Institution"].notna() & df["wiki_matched_title"].notna()])
                st.caption(f"📖 {both_wiki_count:,} with Wikipedia pages")
    
    # === Awards and Fellowships ===
    if "wiki_awards" in df.columns:
        st.subheader("🏆 Recognition")
        col1, col2 = st.columns(2)
        
        with col1:
            df["num_awards"] = df["wiki_awards"].fillna("").apply(lambda x: len(extract_prestigious_awards(x)))
            num_award_winners = (df["num_awards"] > 0).sum()
            st.metric("🏅 Award Winners", num_award_winners)
        
        with col2:
            fellowship_column = "wiki_fellowships" if "wiki_fellowships" in df.columns else "wiki_awards"
            df["num_fellowships"] = df[fellowship_column].fillna("").apply(lambda x: len(extract_prestigious_fellowships(x)))
            num_fellowship_holders = (df["num_fellowships"] > 0).sum()
            st.metric("🎓 Fellowship Holders", num_fellowship_holders)
    
    # === Country Distribution ===
    if "country" in df.columns:
        st.subheader("🌍 Geographic Distribution")
        # Filter out unwanted values
        filtered_countries = df["country"].dropna()
        filtered_countries = filtered_countries[~filtered_countries.str.lower().isin(['unknown', 'academic institution', 'international'])]
        country_counts = filtered_countries.value_counts().head(10)
        
        if not country_counts.empty:
            fig = px.bar(
                x=country_counts.values,
                y=country_counts.index,
                orientation='h',
                title="Top 10 Countries by Number of Researchers",
                labels={'x': 'Number of Researchers', 'y': 'Country'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # === Data Table ===
    st.subheader("📋 Profile Data")
    display_cols = get_display_columns(df)
    
    # Add filters
    col1, col2 = st.columns(2)
    with col1:
        if "country" in df.columns:
            # Filter out unwanted countries from dropdown
            valid_countries = df["country"].dropna()
            valid_countries = valid_countries[~valid_countries.str.lower().isin(['unknown', 'academic institution', 'international'])]
            countries = ["All"] + sorted(valid_countries.unique().tolist())
            country_filter = st.selectbox("Filter by Country", countries, key="overview_country")
            if country_filter != "All":
                df = df[df["country"] == country_filter]
    
    with col2:
        show_count = st.number_input("Show top N rows", min_value=10, max_value=1000, value=100, step=10)
    
    # Display filtered data
    if not df.empty:
        # Replace co-author user IDs with names if coauthors column exists
        if "coauthors" in df.columns:
            userid_to_name = build_userid_to_name_map(df)
            df_display = df.copy()
                        
            # Replace the coauthors column content with actual names
            def convert_coauthors_to_names(coauthors_str):
                names = coauthor_names_from_str(coauthors_str, userid_to_name)
                if names:
                    return ", ".join(names)
                return str(coauthors_str)  # Return original if no names found
            
            df_display["coauthors"] = df_display["coauthors"].apply(convert_coauthors_to_names)
            
            # Reorder columns to put coauthors in the middle
            display_cols_list = list(display_cols)
            if "coauthors" in display_cols_list:
                # Remove coauthors from its current position
                display_cols_list.remove("coauthors")
            
            # Insert coauthors in the middle
            middle_index = len(display_cols_list) // 2
            display_cols_list.insert(middle_index, "coauthors")
                
            st.dataframe(df_display[display_cols_list].head(show_count), use_container_width=True)
        else:
            st.dataframe(df[display_cols].head(show_count), use_container_width=True)
    else:
        st.info("No data to display with current filters.")

def show_search():
    """Display search functionality"""
    st.header("🔍 Search Authors")
    
    df = get_profiles_df()
    if df.empty:
        st.error("No profiles data available for search.")
        return
    
    # === Search Input ===
    search_query = st.text_input(
        "🔎 Enter author name",
        placeholder="e.g., Geoffrey Hinton"
    )
    
    if search_query:
        # Search by name first
        matches_df = fuzzy_author_search(df, search_query, score_cutoff=70)
        
        # If no matches by name, try institution search
        if matches_df.empty:
            matches_df = fuzzy_institution_search(df, search_query, score_cutoff=70)
            search_type = "institution"
        else:
            search_type = "name"
        
        if matches_df.empty:
            st.warning(f"No authors found matching '{search_query}'")
            return
        
        st.success(f"Found {len(matches_df)} author(s) matching '{search_query}' by {search_type}")
        
        # === Filters ===
        with st.expander("🔧 Advanced Filters", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Country filter
                if "country" in matches_df.columns:
                    # Filter out unwanted countries from dropdown
                    valid_countries = matches_df["country"].dropna()
                    valid_countries = valid_countries[~valid_countries.str.lower().isin(['unknown', 'academic institution', 'international'])]
                    countries = ["All"] + sorted(valid_countries.unique())
                    country_filter = st.selectbox("Country", countries)
                    if country_filter != "All":
                        matches_df = matches_df[matches_df["country"] == country_filter]
            
            with col2:
                # H-index filter
                if "h_index_all" in matches_df.columns:
                    try:
                        h_values = matches_df["h_index_all"].dropna().astype(float)
                        if not h_values.empty:
                            min_h, max_h = int(h_values.min()), int(h_values.max())
                            h_range = st.slider("H-Index Range", min_h, max_h, (min_h, max_h))
                            matches_df = matches_df[
                                (matches_df["h_index_all"].astype(float) >= h_range[0]) &
                                (matches_df["h_index_all"].astype(float) <= h_range[1])
                            ]
                    except:
                        st.warning("Could not filter by H-index")
            
            with col3:
                # Sort options
                sort_options = ["Search Relevance (Levenshtein)", "Name", "H-Index", "Citations"]
                sort_by = st.selectbox("Sort by", sort_options)
                sort_order = st.radio("Order", ["Ascending", "Descending"], horizontal=True)
                
                sort_mapping = {
                    "Name": "name",
                    "H-Index": "h_index_all",
                    "Citations": "citations_all"
                }
                
                if sort_by == "Search Relevance (Levenshtein)":
                    # Sort by fuzzy match score (already sorted by default from fuzzy search)
                    pass  # matches_df is already sorted by relevance
                elif sort_by in sort_mapping and sort_mapping[sort_by] in matches_df.columns:
                    matches_df = matches_df.sort_values(
                        sort_mapping[sort_by],
                        ascending=(sort_order == "Ascending"),
                        na_position="last"
                    )
        
        # === Display Results ===
        if matches_df.empty:
            st.info("No results after applying filters.")
        else:
            st.subheader(f"Search Results ({len(matches_df)} authors)")
            
            # Display format options
            display_format = st.radio("Display Format", ["Cards", "Table"], horizontal=True)
            
            if display_format == "Cards":
                # Display as cards
                for idx, row in matches_df.iterrows():
                    with st.expander(f"👤 {row.get('name', 'Unknown')}"):
                        display_author_details(row)
            else:
                # Display as table - show ALL data
                userid_to_name = build_userid_to_name_map(df)
                matches_df = matches_df.copy()
                if "coauthors" in matches_df.columns:
                    matches_df["Co-author Names"] = matches_df["coauthors"].apply(lambda x: ", ".join(coauthor_names_from_str(x, userid_to_name)))
                display_cols = get_display_columns(matches_df)
                if "Co-author Names" in matches_df.columns:
                    display_cols.append("Co-author Names")
                st.dataframe(matches_df[display_cols], use_container_width=True, height=600)
    # Add co-author names column for display if coauthors column exists
    if "coauthors" in df.columns:
        userid_to_name = build_userid_to_name_map(df)
        df = df.copy()
        df["Co-author Names"] = df["coauthors"].apply(lambda x: ", ".join(coauthor_names_from_str(x, userid_to_name)))

    # Download option: only show if matches_df is defined and not empty
    if 'matches_df' in locals() and not matches_df.empty:
        csv = matches_df.to_csv(index=False)
        st.download_button(
            "📥 Download Search Results",
            csv,
            file_name=f"search_results_{search_query.replace(' ', '_')}.csv",
            mime="text/csv"
        )

def display_author_details(author_row):
    """Display detailed information about an author"""
    # Get all available columns
    all_columns = author_row.keys()
    
    # Define excluded columns that we don't want to display
    excluded_fields = {
        "interest_phrases", "wiki_birth_name", "wiki_matched_title", 
        "prestigious_awards_count", "prestigious_fellowships_count"
    }
    
    # === Header with name ===
    name = author_row.get("name", "Unknown")
    st.markdown(f"### 👤 {name}")
    
    # === Main sections in columns ===
    col1, col2 = st.columns([3, 2])
    userid_to_name = build_userid_to_name_map(get_profiles_df())
    with col1:
        st.markdown("#### 📋 Basic Information")
        # Display basic info first
        basic_fields = ["institution", "country", "affiliation", "email", "scholar_id", "user_id"]
        for field in basic_fields:
            if field in author_row and author_row.get(field) not in [None, "", "nan", np.nan]:
                value = author_row.get(field)
                if pd.notna(value) and str(value).strip():
                    label = field.replace("_", " ").title()
                    st.markdown(f"**{label}:** {value}")
        # Academic metrics
        st.markdown("#### 📊 Academic Metrics")
        academic_fields = ["h_index_all", "h_index_5y", "citations_all", "citations_5y", "total_papers", "num_coauthors"]
        for field in academic_fields:
            if field in author_row and author_row.get(field) not in [None, "", "nan", np.nan]:
                value = author_row.get(field)
                if pd.notna(value) and str(value).strip():
                    label = field.replace("_", " ").title()
                    if isinstance(value, (int, float)) and not pd.isna(value):
                        if value >= 1000:
                            formatted_value = f"{value:,.0f}"
                        else:
                            formatted_value = str(value)
                    else:
                        formatted_value = str(value)
                    st.markdown(f"**{label}:** {formatted_value}")
        # Co-authors (show names)
        if "coauthors" in author_row and author_row["coauthors"] not in [None, "", "nan", np.nan]:
            coauthor_names = coauthor_names_from_str(author_row["coauthors"], userid_to_name)
            if coauthor_names:
                st.markdown(f"**Co-authors:** {', '.join(coauthor_names)}")
    
    with col2:
        # Conference participation
        st.markdown("#### 🎯 Conference Participation")
        conferences = []
        conference_fields = ["NeurIPS_Institution", "ICLR_Institution", "AAAI_Institution"]
        
        for field in conference_fields:
            if author_row.get(field) and pd.notna(author_row.get(field)):
                conf_name = field.replace("_Institution", "")
                if conf_name == "NeurIPS":
                    conferences.append("🧠 NeurIPS")
                elif conf_name == "ICLR":
                    conferences.append("📖 ICLR")
                elif conf_name == "AAAI":
                    conferences.append("🤖 AAAI")
                
                # Show institution for this conference
                institution = author_row.get(field)
                st.markdown(f"**{conf_name}:** {institution}")
        
        if not conferences:
            st.markdown("*No conference participation recorded*")
    
    # === Research Areas and Interests ===
    research_fields = ["interests", "topics", "research_areas", "keywords"]
    research_content = []
    for field in research_fields:
        if field in author_row and author_row.get(field) not in [None, "", "nan", np.nan]:
            value = author_row.get(field)
            if pd.notna(value) and str(value).strip():
                research_content.append(str(value))
    
    if research_content:
        st.markdown("#### 🔬 Research Areas & Interests")
        for content in research_content:
            # Try to parse if it's a list string
            try:
                if content.startswith('[') and content.endswith(']'):
                    parsed_content = ast.literal_eval(content)
                    if isinstance(parsed_content, list):
                        st.markdown("- " + ", ".join(str(item) for item in parsed_content))
                    else:
                        st.markdown(f"- {content}")
                else:
                    st.markdown(f"- {content}")
            except:
                st.markdown(f"- {content}")
    
    # === Awards and Recognition ===
    awards_str = author_row.get("wiki_awards", "")
    fellowship_str = author_row.get("wiki_fellowships", "") or awards_str
    
    if awards_str or fellowship_str:
        st.markdown("#### 🏆 Awards & Recognition")
        
        # Show prestigious awards and fellowships with colored tags
        prestigious_awards = extract_prestigious_awards(awards_str)
        prestigious_fellowships = extract_prestigious_fellowships(fellowship_str)
        
        if prestigious_awards or prestigious_fellowships:
            all_honors = prestigious_awards + prestigious_fellowships
            honor_tags = render_colored_tags(all_honors)
            st.markdown(honor_tags, unsafe_allow_html=True)
        
        # Show full awards text if available
        if awards_str and str(awards_str).strip() not in ["", "nan", "None"]:
            with st.expander("View Full Awards & Recognition Text"):
                st.text(awards_str)
    
    # === Additional Information ===
    st.markdown("#### 📄 Additional Information")
    
    # Show all remaining fields that haven't been displayed yet
    displayed_fields = {
        "name", "institution", "country", "affiliation", "email", "scholar_id", "user_id",
        "h_index_all", "h_index_5y", "citations_all", "citations_5y", "total_papers", "num_coauthors",
        "NeurIPS_Institution", "ICLR_Institution", "AAAI_Institution",
        "interests", "topics", "research_areas", "keywords", "wiki_awards", "wiki_fellowships"
    } | excluded_fields
    
    additional_info = []
    for field in all_columns:
        if field not in displayed_fields and field in author_row:
            value = author_row.get(field)
            if value not in [None, "", "nan", np.nan] and pd.notna(value) and str(value).strip():
                label = field.replace("_", " ").title()
                # Handle different data types
                if isinstance(value, (list, dict)):
                    try:
                        value_str = str(value)
                        if len(value_str) > 100:
                            additional_info.append((label, f"{value_str[:100]}... (truncated)"))
                        else:
                            additional_info.append((label, value_str))
                    except:
                        additional_info.append((label, "Complex data structure"))
                else:
                    value_str = str(value)
                    if len(value_str) > 200:
                        additional_info.append((label, f"{value_str[:200]}... (truncated)"))
                    else:
                        additional_info.append((label, value_str))
    
    if additional_info:
        for label, value in additional_info:
            st.markdown(f"**{label}:** {value}")
    else:
        st.markdown("*No additional information available*")
    
    # === Raw Data Toggle ===
    with st.expander("🔍 View Raw Data (All Fields)"):
        # Create a clean dictionary with all non-null values
        clean_data = {}
        for field in all_columns:
            value = author_row.get(field)
            if value not in [None, "", "nan", np.nan] and pd.notna(value):
                clean_data[field] = value
        
        st.json(clean_data)

def count_singaporean_coauthors(coauthors_str, singaporean_user_ids):
    """Count how many co-authors are from Singapore using pre-collected Singaporean user IDs"""
    if pd.isna(coauthors_str) or not coauthors_str:
        return 0
    
    try:
        if isinstance(coauthors_str, str) and coauthors_str.startswith('['):
            coauthor_list = ast.literal_eval(coauthors_str)
            singapore_count = 0
            
            for coauthor in coauthor_list:
                if isinstance(coauthor, dict):
                    coauthor_user_id = coauthor.get("user_id", "")
                    if coauthor_user_id in singaporean_user_ids:
                        singapore_count += 1
                else:
                    # Handle cases where coauthor might be just a user ID string
                    coauthor_str = str(coauthor)
                    if coauthor_str in singaporean_user_ids:
                        singapore_count += 1
            
            return singapore_count
    except:
        return 0
    
    return 0

def show_analytics():
    """Display analytics and visualizations"""
    st.header("📈 Analytics Dashboard")
    
    df = get_profiles_df()
    if df.empty:
        st.error("No data available for analytics.")
        return

    # === Singaporean Co-authors Analysis ===
    if "coauthors" in df.columns and "user_id" in df.columns:
        userid_to_name = build_userid_to_name_map(df)
        st.subheader("🇸🇬 Singaporean Co-authors Analysis")
        
        with st.spinner("Calculating Singaporean co-author counts..."):
            # First, collect all Singaporean user IDs for efficient lookup
            singaporean_researchers = df[df["country"].str.lower() == "singapore"]
            singaporean_user_ids = set(singaporean_researchers["user_id"].dropna().astype(str))
            
            st.info(f"Found {len(singaporean_user_ids):,} Singaporean researchers in database")
            
            # Calculate Singaporean co-author counts for each researcher
            df_copy = df.copy()
            df_copy["singaporean_coauthors_count"] = df_copy["coauthors"].apply(
                lambda x: count_singaporean_coauthors(x, singaporean_user_ids)
            )
            
            # Display statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_with_sg_coauthors = (df_copy["singaporean_coauthors_count"] > 0).sum()
                st.metric("Researchers with SG Co-authors", f"{total_with_sg_coauthors:,}")
            
            with col2:
                avg_sg_coauthors = df_copy["singaporean_coauthors_count"].mean()
                st.metric("Avg SG Co-authors", f"{avg_sg_coauthors:.2f}")
            
            with col3:
                max_sg_coauthors = df_copy["singaporean_coauthors_count"].max()
                st.metric("Max SG Co-authors", max_sg_coauthors)
            
            with col4:
                total_sg_connections = df_copy["singaporean_coauthors_count"].sum()
                st.metric("Total SG Connections", f"{total_sg_connections:,}")
            
            # Show top researchers with most Singaporean co-authors
            col1, col2 = st.columns([3, 1])
            with col1:
                st.subheader("🔝 Top Researchers by Singaporean Co-authors")
            with col2:
                # Add number input for controlling how many results to show
                show_sg_count = st.number_input(
                    "Show top N researchers",
                    min_value=5,
                    max_value=100,
                    value=20,
                    step=5,
                    help="Choose how many top researchers to display"
                )
                exclude_singaporeans = st.checkbox(
                    "📍 Non-Singaporeans only",
                    value=False,
                    help="Show only researchers from outside Singapore who collaborate with Singaporean researchers"
                )

            # Filter data based on checkbox
            if exclude_singaporeans:
                # Filter out researchers who are from Singapore
                top_sg_researchers = df_copy[
                    (df_copy["singaporean_coauthors_count"] > 0) &
                    (df_copy["country"].str.lower() != "singapore")
                ].nlargest(show_sg_count, "singaporean_coauthors_count")

                if not top_sg_researchers.empty:
                    st.info(f"Showing top {len(top_sg_researchers)} non-Singaporean researchers with Singaporean collaborations")
                else:
                    st.warning("No non-Singaporean researchers found with Singaporean co-authors")
            else:
                # Show all researchers (including Singaporeans)
                top_sg_researchers = df_copy[df_copy["singaporean_coauthors_count"] > 0].nlargest(show_sg_count, "singaporean_coauthors_count")

                if not top_sg_researchers.empty:
                    st.info(f"Showing top {len(top_sg_researchers)} researchers (all countries) with Singaporean collaborations")
            
            if not top_sg_researchers.empty:
                
                display_cols = ["name", "institution", "country", "singaporean_coauthors_count"]
                if "h_index_all" in df_copy.columns:
                    display_cols.append("h_index_all")
                # Add coauthor names column for display
                top_sg_researchers = top_sg_researchers.copy()
                top_sg_researchers["SG Co-author Names"] = top_sg_researchers["coauthors"].apply(
                    lambda x: ", ".join(coauthor_names_from_str(x, userid_to_name))
                )
                # Filter to only existing columns
                available_cols = [col for col in display_cols if col in top_sg_researchers.columns]
                # Insert coauthor names column after count
                if "singaporean_coauthors_count" in available_cols:
                    idx = available_cols.index("singaporean_coauthors_count")
                    available_cols.insert(idx+1, "SG Co-author Names")
                st.dataframe(
                    top_sg_researchers[available_cols].rename(columns={
                        "singaporean_coauthors_count": "SG Co-authors",
                        "h_index_all": "H-Index"
                    }),
                    use_container_width=True
                )
                
                # Visualization: Distribution of Singaporean co-authors
                st.subheader("📊 Distribution of Singaporean Co-authors")
                
                # Create histogram
                sg_counts = df_copy["singaporean_coauthors_count"]
                sg_counts_filtered = sg_counts[sg_counts > 0]  # Only show those with at least 1
                
                if not sg_counts_filtered.empty:
                    # Convert to regular Python int to avoid numpy.int64 issues
                    max_count = int(sg_counts_filtered.max())
                    nbins = min(20, max_count)
                    
                    fig = px.histogram(
                        sg_counts_filtered,
                        nbins=nbins,
                        title="Distribution of Singaporean Co-authors Count",
                        labels={'value': 'Number of Singaporean Co-authors', 'count': 'Number of Researchers'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show some additional statistics
                    st.write("**Summary Statistics:**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Median", int(sg_counts_filtered.median()))
                    with col2:
                        st.metric("Most Common", int(sg_counts_filtered.mode().iloc[0]) if not sg_counts_filtered.mode().empty else "N/A")
                    with col3:
                        st.metric("95th Percentile", int(sg_counts_filtered.quantile(0.95)))
                else:
                    st.info("No researchers found with Singaporean co-authors.")
            else:
                st.info("No researchers found with Singaporean co-authors.")
    elif "coauthors" not in df.columns:
        st.warning("No co-author data available for analysis.")
    elif "user_id" not in df.columns:
        st.warning("No user ID data available for analysis.")

    # === H-Index Distribution ===
    if "h_index_all" in df.columns:
        st.subheader("📊 H-Index Distribution")
        h_values = df["h_index_all"].dropna().astype(float)
        if not h_values.empty:
            fig = px.histogram(
                h_values,
                nbins=50,
                title="Distribution of H-Index Across All Researchers",
                labels={'value': 'H-Index', 'count': 'Number of Researchers'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
   


def show_network():
    """Display network visualization options and generate networks""" 
    st.header("🌐 Author Network Visualization")
    
    df = get_profiles_df()
    if df.empty:
        st.error("No data available for network analysis.")
        return
    
    # Check if we have coauthor data
    if "coauthors" not in df.columns:
        st.warning("No coauthor data available for network analysis.")
        return
    
    # === Network Options ===
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔗 Generate Collaboration Network")
        
        # Network type selection
        network_type = st.selectbox(
            "Network Type",
            ["Author Collaboration"],
            help="Choose the type of network to visualize"
        )

        if network_type == "Author Collaboration":
            # Author search for network center
            search_query = st.text_input(
                "Search for author to center network around",
                placeholder="e.g., Geoffrey Hinton",
                help="The network will be built around this author and their collaborators"
            )
            
            if search_query:
                # Find matching authors
                matches = df[df["name"].str.contains(search_query, case=False, na=False)]
                if not matches.empty:
                    # Create detailed options for author selection
                    author_options = []
                    author_details = {}
                    
                    for idx, row in matches.iterrows():
                        name = row.get("name", "Unknown")
                        institution = row.get("institution", "Unknown")
                        country = row.get("country", "Unknown")
                        h_index = row.get("h_index_all", "N/A")
                        citations = row.get("citations_all", "N/A")
                        
                        # Format citations with commas
                        if pd.notna(citations) and citations != "N/A":
                            try:
                                citations_formatted = f"{int(float(citations)):,}"
                            except:
                                citations_formatted = str(citations)
                        else:
                            citations_formatted = "N/A"
                        
                        # Create a detailed display option
                        option_text = f"{name} | {institution} | {country} | H-index: {h_index} | Citations: {citations_formatted}"
                        author_options.append(option_text)
                        author_details[option_text] = name
                    
                    # Show author details for selection
                    if len(author_options) > 1:
                        st.info(f"Found {len(author_options)} researchers matching '{search_query}'. Please select one:")
                        
                        # Display options in an expander for better visibility
                        with st.expander("📋 Author Comparison", expanded=True):
                            comparison_df = matches[["name", "institution", "country", "h_index_all", "citations_all"]].copy()
                            comparison_df.columns = ["Name", "Institution", "Country", "H-Index", "Citations"]
                            
                            # Format citations column
                            comparison_df["Citations"] = comparison_df["Citations"].apply(
                                lambda x: f"{int(float(x)):,}" if pd.notna(x) and x != "N/A" else "N/A"
                            )
                            
                            st.dataframe(comparison_df, use_container_width=True)
                    
                    selected_option = st.selectbox(
                        "Select Author (Name | Institution | Country | H-index | Citations)",
                        author_options,
                        help="Choose the specific author from search results"
                    )
                    
                    selected_author = author_details[selected_option]
                    
                    degree_choice = st.selectbox(
                        "Degrees of Separation",
                        [1, 2, 3],
                        index=0,
                        help="How many degrees of separation to include"
                    )
                    
                    if st.button("🚀 Generate Author Network"):
                        generate_author_network(df, selected_author, degree_choice)
                else:
                    st.warning("No authors found matching your search.")
    
    with col2:
        st.subheader("📊 Network Statistics")
        
        # Basic network stats
        if "coauthors" in df.columns:
            total_researchers = len(df)
            researchers_with_coauthors = df["coauthors"].notna().sum()
            
            st.metric("Total Researchers", f"{total_researchers:,}")
            
            # Additional stats
            if researchers_with_coauthors > 0:
                # Calculate average number of coauthors
                coauthor_counts = []
                for idx, row in df.iterrows():
                    coauthors = row.get("coauthors")
                    if pd.notna(coauthors) and coauthors:
                        try:
                            if isinstance(coauthors, str) and coauthors.startswith('['):
                                coauthor_list = ast.literal_eval(coauthors)
                                coauthor_counts.append(len(coauthor_list))
                        except:
                            pass
                
                if coauthor_counts:
                    avg_coauthors = np.mean(coauthor_counts)
                    st.metric("Avg Co-authors", f"{avg_coauthors:.1f}")
                    st.metric("Max Co-authors", max(coauthor_counts))

def calculate_node_sizes(G, min_size=200, max_size=1500, base_size=400):
    """Calculate node sizes based on H-index values"""
    h_indices = []
    for node in G.nodes():
        h_index = G.nodes[node].get('h_index', 0)
        h_indices.append(h_index)
    
    if not h_indices or all(h == 0 for h in h_indices):
        # If no H-index data, use default sizes based on level
        node_sizes = []
        for node in G.nodes():
            level = G.nodes[node].get('level', 0)
            if level == 0:
                node_sizes.append(1200)  # Central author
            elif level == 1:
                node_sizes.append(800)   # Direct collaborators
            elif level == 2:
                node_sizes.append(500)   # 2nd degree
            else:
                node_sizes.append(300)   # Extended network
        return node_sizes
    
    # Calculate sizes based on H-index
    min_h = min(h_indices)
    max_h = max(h_indices)
    
    # Avoid division by zero
    if max_h == min_h:
        return [base_size] * len(G.nodes())
    
    node_sizes = []
    for node in G.nodes():
        h_index = G.nodes[node].get('h_index', 0)
        # Scale H-index to size range, with minimum size for visibility
        if h_index == 0:
            size = min_size
        else:
            # Logarithmic scaling for better visual differentiation
            size = min_size + (max_size - min_size) * (h_index - min_h) / (max_h - min_h)
            # Ensure minimum size
            size = max(size, min_size)
        node_sizes.append(size)
    
    return node_sizes

def generate_author_network(df, selected_author, degrees):
    """Generate and display author collaboration network"""
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
        import io
        import base64
        import time

        st.subheader(f"🤝 Collaboration Network for {selected_author}")

        # Start timing for performance monitoring
        start_time = time.time()

        # Check if author exists
        if selected_author not in df["name"].values:
            st.error(f"Author '{selected_author}' not found in the dataset.")
            return

        # Create efficient lookup dictionaries for better performance
        st.info("🔄 Building author lookup tables...")
        name_to_data = {}
        name_to_coauthors = {}
        user_id_to_name = {}

        for idx, row in df.iterrows():
            name = row.get("name", "")
            user_id = row.get("user_id", "")
            if name:
                name_to_data[name] = row
                if user_id:
                    user_id_to_name[user_id] = name

        # Second pass to build coauthor relationships
        for idx, row in df.iterrows():
            name = row.get("name", "")
            coauthors = row.get("coauthors", "")
            if name and pd.notna(coauthors) and coauthors:
                try:
                    if isinstance(coauthors, str) and coauthors.startswith('['):
                        coauthor_list = ast.literal_eval(coauthors)
                        # Convert user IDs to names
                        coauthor_names = []
                        for coauthor_id in coauthor_list:
                            if isinstance(coauthor_id, str) and coauthor_id in user_id_to_name:
                                coauthor_names.append(user_id_to_name[coauthor_id])
                            elif isinstance(coauthor_id, dict) and coauthor_id.get("user_id") in user_id_to_name:
                                coauthor_names.append(user_id_to_name[coauthor_id.get("user_id")])
                        name_to_coauthors[name] = coauthor_names[:20]  # Limit to prevent explosion
                    elif isinstance(coauthors, list):
                        coauthor_names = []
                        for coauthor_id in coauthors:
                            if isinstance(coauthor_id, str) and coauthor_id in user_id_to_name:
                                coauthor_names.append(user_id_to_name[coauthor_id])
                        name_to_coauthors[name] = coauthor_names[:20]
                except Exception as e:
                    # Skip problematic coauthor data
                    continue

        st.info(f"✅ Processed {len(name_to_data)} authors and {len(user_id_to_name)} user mappings for network analysis")

        # Create network graph
        G = nx.Graph()

        # Add central author
        central_author_data = name_to_data.get(selected_author)
        central_h_index = 0
        if central_author_data is not None:
            try:
                h_index_val = central_author_data.get("h_index_all")
                if pd.notna(h_index_val) and h_index_val != "N/A":
                    central_h_index = float(h_index_val)
            except:
                central_h_index = 0
        
        G.add_node(selected_author, name=selected_author, type="central", level=0, h_index=central_h_index)

        # Build network with proper limits to prevent infinite loops
        added_nodes = {selected_author}
        current_level_nodes = [(selected_author, 0)]
        max_nodes = 20000  # Hard limit to prevent memory issues
        max_edges_per_node = 21  # Limit connections per node

        for level in range(degrees):
            if len(G.nodes()) >= max_nodes:
                st.warning(f"⚠️ Network size limit reached ({max_nodes} nodes). Stopping expansion.")
                break

            next_level_nodes = []
            level_start_time = time.time()

            st.info(f"🔄 Processing level {level + 1} connections...")

            for node_name, current_level in current_level_nodes:
                if node_name in name_to_coauthors:
                    coauthor_names = name_to_coauthors[node_name]

                    # Limit coauthors per node to prevent network explosion
                    coauthor_names = coauthor_names[:max_edges_per_node]

                    for coauthor_name in coauthor_names:
                        if coauthor_name and coauthor_name != node_name:
                            # Check if this coauthor exists in our dataset
                            if coauthor_name in name_to_data:
                                if coauthor_name not in added_nodes:
                                    if len(G.nodes()) < max_nodes:
                                        # Get H-index for this coauthor
                                        coauthor_data = name_to_data.get(coauthor_name)
                                        coauthor_h_index = 0
                                        if coauthor_data is not None:
                                            try:
                                                h_index_val = coauthor_data.get("h_index_all")
                                                if pd.notna(h_index_val) and h_index_val != "N/A":
                                                    coauthor_h_index = float(h_index_val)
                                            except:
                                                coauthor_h_index = 0
                                        
                                        G.add_node(coauthor_name, name=coauthor_name,
                                                 type="collaborator", level=current_level+1, h_index=coauthor_h_index)
                                        added_nodes.add(coauthor_name)
                                        next_level_nodes.append((coauthor_name, current_level+1))

                                # Add edge (only if both nodes exist)
                                if coauthor_name in added_nodes:
                                    G.add_edge(node_name, coauthor_name)

            current_level_nodes = next_level_nodes

            level_time = time.time() - level_start_time
            st.info(f"✅ Level {level + 1} completed in {level_time:.2f}s. Added {len(next_level_nodes)} new nodes.")

            if not next_level_nodes:
                st.info("🔄 No more connections to expand.")
                break

        total_time = time.time() - start_time
        node_count = len(G.nodes())
        edge_count = len(G.edges())

        st.success(f"🎉 Network built in {total_time:.2f}s! Found {node_count} researchers with {edge_count} connections.")

        if node_count <= 1:
            st.warning("No collaboration data found for this author.")
            return

        # Always provide Gephi download option regardless of network size
        try:
            gexf_content = generate_gephi_file(G, selected_author)

            # Create download button - always available
            st.download_button(
                label="📥 Download Gephi Network File (.gexf)",
                data=gexf_content,
                file_name=f"author_network_{selected_author.replace(' ', '_')}.gexf",
                mime="application/xml",
                help="Download this file and open it in Gephi for advanced network analysis and visualization",
                key="gephi_download"
            )

        except Exception as e:
            st.error(f"Error generating Gephi file: {str(e)}")

        # Check node limit for visualization
        NODE_LIMIT = 200

        if node_count > NODE_LIMIT:
            st.warning(f"⚠️ Network too large for optimal dashboard visualization ({node_count} nodes, limit: {NODE_LIMIT})")
            st.info("💡 The Gephi download above provides full network analysis capabilities")

            # Still show basic statistics even for large networks
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Nodes", node_count)
            with col2:
                st.metric("Total Connections", edge_count)
            with col3:
                if node_count > 1:
                    density = nx.density(G)
                    st.metric("Network Density", f"{density:.3f}")

            # Show degree breakdown
            level_counts = {}
            for node in G.nodes():
                level = G.nodes[node].get('level', 0)
                level_counts[level] = level_counts.get(level, 0) + 1

            st.write("**Network Breakdown by Degree:**")
            for level in sorted(level_counts.keys()):
                if level == 0:
                    st.write(f"- Central Author: {level_counts[level]} researcher")
                else:
                    degree_text = "1st" if level == 1 else "2nd" if level == 2 else f"{level}th"
                    st.write(f"- {degree_text} Degree: {level_counts[level]} researchers")

        else:
            # Display network visualization in dashboard
            st.info(f"📊 Displaying network with {node_count} nodes (within limit of {NODE_LIMIT})")

            try:
                # Create visualization with better layout and styling
                fig, ax = plt.subplots(figsize=(14, 10))
                fig.patch.set_facecolor('#f8f9fa')  # Light background

                # Use better layout algorithms for cleaner visualization
                if node_count <= 20:
                    # For small networks, use spring layout with better spacing
                    pos = nx.spring_layout(G, k=2.5, iterations=100, seed=42)
                elif node_count <= 50:
                    # For medium networks, use Kamada-Kawai layout (cleaner)
                    pos = nx.kamada_kawai_layout(G)
                else:
                    # For larger networks, use Fruchterman-Reingold (good balance)
                    pos = nx.fruchterman_reingold_layout(G, k=1.5, iterations=100, seed=42)

                # Enhanced node styling with H-index-based sizes
                node_colors = []
                node_sizes = calculate_node_sizes(G)  # Use H-index for sizing
                node_alphas = []

                for node in G.nodes():
                    level = G.nodes[node].get('level', 0)
                    if level == 0:
                        node_colors.append('#DC143C')  # Crimson red for central author
                        node_alphas.append(0.95)
                    elif level == 1:
                        node_colors.append('#4169E1')  # Royal blue for direct collaborators
                        node_alphas.append(0.85)
                    elif level == 2:
                        node_colors.append('#32CD32')  # Lime green for 2nd degree
                        node_alphas.append(0.75)
                    else:
                        node_colors.append('#FFA500')  # Orange for extended network
                        node_alphas.append(0.65)

                # Draw edges with varying thickness based on connection strength
                edge_colors = []
                edge_widths = []
                for u, v in G.edges():
                    level_u = G.nodes[u].get('level', 0)
                    level_v = G.nodes[v].get('level', 0)
                    if level_u == 0 or level_v == 0:
                        edge_colors.append('#FF6B6B')  # Red for central connections
                        edge_widths.append(2.5)
                    elif max(level_u, level_v) == 1:
                        edge_colors.append('#87CEEB')  # Sky blue for 1st degree connections
                        edge_widths.append(2.0)
                    else:
                        edge_colors.append('#D3D3D3')  # Light gray for extended connections
                        edge_widths.append(1.5)

                # Draw the network
                # Draw edges with different styles for different connection types
                central_edges = [(u, v) for u, v in G.edges() if G.nodes[u].get('level', 0) == 0 or G.nodes[v].get('level', 0) == 0]
                first_degree_edges = [(u, v) for u, v in G.edges() if max(G.nodes[u].get('level', 0), G.nodes[v].get('level', 0)) == 1 and (u, v) not in central_edges]
                extended_edges = [(u, v) for u, v in G.edges() if (u, v) not in central_edges and (u, v) not in first_degree_edges]

                if central_edges:
                    nx.draw_networkx_edges(G, pos, edgelist=central_edges, edge_color='#FF6B6B',
                                         width=2.5, alpha=0.7, ax=ax, arrows=False)
                if first_degree_edges:
                    nx.draw_networkx_edges(G, pos, edgelist=first_degree_edges, edge_color='#87CEEB',
                                         width=2.0, alpha=0.6, ax=ax, arrows=False)
                if extended_edges:
                    nx.draw_networkx_edges(G, pos, edgelist=extended_edges, edge_color='#D3D3D3',
                                         width=1.5, alpha=0.5, ax=ax, arrows=False)

                # Draw nodes with H-index-based sizing (individual node approach)
                # Create node size mapping
                node_size_map = dict(zip(G.nodes(), node_sizes))

                # Draw each node individually with its H-index-based size
                for node in G.nodes():
                    level = G.nodes[node].get('level', 0)
                    node_size = node_size_map[node]

                    # Set color based on level
                    if level == 0:
                        color = '#DC143C'  # Crimson red for central author
                        alpha = 0.95
                    elif level == 1:
                        color = '#4169E1'  # Royal blue for direct collaborators
                        alpha = 0.85
                    elif level == 2:
                        color = '#32CD32'  # Lime green for 2nd degree
                        alpha = 0.75
                    else:
                        color = '#FFA500'  # Orange for extended network
                        alpha = 0.65

                    # Draw individual node
                    nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color=color,
                                         node_size=node_size, alpha=alpha, ax=ax,
                                         edgecolors='white', linewidths=2)

                # Simplified labeling system - show ALL co-author names regardless of network size
                labels = {}
                for node in G.nodes():
                    name = G.nodes[node].get('name', str(node))
                    level = G.nodes[node].get('level', 0)

                    # Format names based on collaboration level with visual indicators
                    if level == 0:
                        # Central author - highlighted with star
                        if len(name) > 35:
                            name = name[:32] + "..."
                        labels[node] = f"⭐ {name}"
                    elif level == 1:
                        # Direct collaborators - full names
                        if len(name) > 30:
                            name = name[:27] + "..."
                        labels[node] = name
                    elif level == 2:
                        # 2nd degree collaborators - with bullet point
                        if len(name) > 25:
                            name = name[:22] + "..."
                        labels[node] = f"• {name}"
                    else:
                        # Extended network - shorter names
                        if len(name) > 20:
                            name = name[:17] + "..."
                        labels[node] = name

                # Draw all labels with consistent styling
                if labels:
                    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold',
                                          font_color='black', ax=ax,
                                          bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

                # Add a note about the comprehensive labeling
                labeling_note = f"📝 Showing all {node_count} co-author names"
                ax.text(0.02, 0.02, labeling_note, transform=ax.transAxes,
                       fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

                # Add title and styling
                plt.title(f'Author Collaboration Network\n{selected_author}\n(Node sizes represent H-index)',
                         fontsize=14, fontweight='bold', pad=20)

                # Add legend with H-index information
                from matplotlib.lines import Line2D
                legend_elements = [
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='#DC143C',
                             markersize=15, label='Central Author'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='#4169E1',
                             markersize=12, label='Direct Collaborators'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='#32CD32',
                             markersize=10, label='2nd Degree Connections'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFA500',
                             markersize=8, label='Extended Network'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray',
                             markersize=6, label='Node size = H-index')
                ]

                ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1),
                         fontsize=10, framealpha=0.9)

                plt.axis('off')
                plt.tight_layout()

                # Display the network
                st.pyplot(fig)

                # Show degree breakdown
                level_counts = {}
                for node in G.nodes():
                    level = G.nodes[node].get('level', 0)
                    level_counts[level] = level_counts.get(level, 0) + 1

                st.write("**Network Breakdown by Degree:**")
                for level in sorted(level_counts.keys()):
                    if level == 0:
                        st.write(f"- Central Author: {level_counts[level]} researcher")
                    else:
                        degree_text = "1st" if level == 1 else "2nd" if level == 2 else f"{level}th"
                        st.write(f"- {degree_text} Degree: {level_counts[level]} researchers")

                # Network statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Nodes", node_count)
                with col2:
                    st.metric("Total Connections", edge_count)
                with col3:
                    if node_count > 1:
                        density = nx.density(G)
                        st.metric("Network Density", f"{density:.3f}")

                # H-index statistics for the network
                h_indices = [G.nodes[node].get('h_index', 0) for node in G.nodes()]
                if h_indices and any(h > 0 for h in h_indices):
                    valid_h_indices = [h for h in h_indices if h > 0]
                    if valid_h_indices:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Avg H-Index", f"{np.mean(valid_h_indices):.1f}")
                        with col2:
                            st.metric("Max H-Index", f"{max(valid_h_indices):.0f}")
                        with col3:
                            st.metric("Min H-Index", f"{min(valid_h_indices):.0f}")
                        
                        st.info("💡 **Node sizes are proportional to H-index values** - larger nodes indicate higher H-index scores")

            except Exception as e:
                st.error(f"Error creating visualization: {str(e)}")
                st.info("Try downloading the Gephi file instead for network analysis.")

    except ImportError:
        st.error("NetworkX and matplotlib are required for network visualization. Please install them:")
        st.code("pip install networkx matplotlib")
    except Exception as e:
        st.error(f"Error generating network: {str(e)}")
        st.info("💡 Try with a different author or reduce the degrees of separation.")

def generate_institution_network(df):
    """Generate institution collaboration network"""
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
        
        st.subheader("🏛️ Institution Collaboration Network")
        
        # Create institution network
        G = nx.Graph()
        institution_collaborations = {}
        
        for idx, row in df.iterrows():
            institution = row.get("institution")
            coauthors = row.get("coauthors")
            
            if pd.notna(institution) and pd.notna(coauthors) and coauthors:
                try:
                    if isinstance(coauthors, str) and coauthors.startswith('['):
                        coauthor_list = ast.literal_eval(coauthors)
                        
                        for coauthor in coauthor_list:
                            # Find coauthor's institution
                            if isinstance(coauthor, dict):
                                coauthor_name = coauthor.get("name", "")
                            else:
                                coauthor_name = str(coauthor)
                            
                            coauthor_data = df[df["name"] == coauthor_name]
                            if not coauthor_data.empty:
                                coauthor_inst = coauthor_data.iloc[0].get("institution")
                                if pd.notna(coauthor_inst) and coauthor_inst != institution:
                                    # Add institutions as nodes
                                    G.add_node(institution)
                                    G.add_node(coauthor_inst)
                                    
                                    # Add or strengthen edge
                                    if G.has_edge(institution, coauthor_inst):
                                        G[institution][coauthor_inst]['weight'] += 1
                                    else:
                                        G.add_edge(institution, coauthor_inst, weight=1)
                except:
                    pass
        
        if len(G.nodes()) > 1:
            # Filter to show only institutions with multiple connections
            min_connections = st.slider("Minimum connections", 1, 10, 2)
            nodes_to_remove = [node for node in G.nodes() if len(G[node]) < min_connections]
            G.remove_nodes_from(nodes_to_remove)
            
            if len(G.nodes()) > 1:
                fig, ax = plt.subplots(figsize=(14, 10))
                
                pos = nx.spring_layout(G, k=2, iterations=50)
                
                # Node sizes based on degree
                node_sizes = [len(G[node]) * 100 for node in G.nodes()]

                # Edge widths based on collaboration strength
                edge_widths = [G[u][v]['weight'] * 0.5 for u, v in G.edges()]

                # Draw nodes and edges with proper styling
                for node, size in zip(G.nodes(), node_sizes):
                    nx.draw_networkx_nodes(G, pos, nodelist=[node], node_size=size,
                                         node_color='lightblue', alpha=0.7)

                for (u, v), width in zip(G.edges(), edge_widths):
                    nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=width, alpha=0.5)
                
                # Add labels
                labels = {node: node[:20] + "..." if len(node) > 20 else node for node in G.nodes()}
                nx.draw_networkx_labels(G, pos, labels, font_size=8)
                
                plt.title(f"Institution Collaboration Network\n({len(G.nodes())} institutions, {len(G.edges())} collaborations)")
                plt.axis('off')
                plt.tight_layout()
                
                st.pyplot(fig)
                
                # Top collaborating institutions
                degree_centrality = nx.degree_centrality(G)
                top_institutions = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
                
                st.subheader("🔝 Most Connected Institutions")
                for i, (inst, centrality) in enumerate(top_institutions, 1):
                    st.write(f"{i}. **{inst}** - Centrality: {centrality:.3f}")
            else:
                st.warning("No institution network found with the current filter settings.")
        else:
            st.warning("Insufficient data to generate institution network.")
            
    except ImportError:
        st.error("NetworkX and matplotlib are required. Please install them:")
        st.code("pip install networkx matplotlib")
    except Exception as e:
        st.error(f"Error generating institution network: {str(e)}")

def generate_country_network(df):
    """Generate country collaboration network"""
    try:
        import networkx as nx
        import matplotlib.pyplot as plt

        st.subheader("🌍 Country Collaboration Network")

        # Filter out unwanted countries
        filtered_df = df[df["country"].notna()]
        filtered_df = filtered_df[~filtered_df["country"].str.lower().isin(['unknown', 'academic institution', 'international'])]

        # Create country network
        G = nx.Graph()

        for idx, row in filtered_df.iterrows():
            country = row.get("country")
            coauthors = row.get("coauthors")

            if pd.notna(country) and pd.notna(coauthors) and coauthors:
                try:
                    if isinstance(coauthors, str) and coauthors.startswith('['):
                        coauthor_list = ast.literal_eval(coauthors)

                        for coauthor in coauthor_list:
                            if isinstance(coauthor, dict):
                                coauthor_name = coauthor.get("name", "")
                            else:
                                coauthor_name = str(coauthor)

                            coauthor_data = df[df["name"] == coauthor_name]
                            if not coauthor_data.empty:
                                coauthor_country = coauthor_data.iloc[0].get("country")
                                if (pd.notna(coauthor_country) and
                                    coauthor_country != country and
                                    coauthor_country.lower() not in ['unknown', 'academic institution', 'international']):

                                    G.add_node(country)
                                    G.add_node(coauthor_country)

                                    if G.has_edge(country, coauthor_country):
                                        G[country][coauthor_country]['weight'] += 1
                                    else:
                                        G.add_edge(country, coauthor_country, weight=1)
                except:
                    pass

        if len(G.nodes()) > 1:
            # Filter to show meaningful connections
            min_connections = st.slider("Minimum international collaborations", 1, 20, 5)
            edges_to_remove = [(u, v) for u, v in G.edges() if G[u][v]['weight'] < min_connections]
            G.remove_edges_from(edges_to_remove)

            # Remove isolated nodes
            isolated_nodes = [node for node in G.nodes() if len(G[node]) == 0]
            G.remove_nodes_from(isolated_nodes)

            if len(G.nodes()) > 1:
                fig, ax = plt.subplots(figsize=(14, 10))

                pos = nx.spring_layout(G, k=3, iterations=50)

                # Node sizes based on total collaborations
                node_sizes = [sum([G[node][neighbor]['weight'] for neighbor in G.neighbors(node)]) * 5 for node in G.nodes()]

                # Edge widths based on collaboration strength
                edge_widths = [G[u][v]['weight'] * 0.1 for u, v in G.edges()]

                # Draw nodes and edges with proper styling
                for node, size in zip(G.nodes(), node_sizes):
                    nx.draw_networkx_nodes(G, pos, nodelist=[node], node_size=size,
                                         node_color='lightgreen', alpha=0.7)

                for (u, v), width in zip(G.edges(), edge_widths):
                    nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=width, alpha=0.6)
                nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

                plt.title(f"International Collaboration Network\n({len(G.nodes())} countries, {len(G.edges())} collaboration pairs)")
                plt.axis('off')
                plt.tight_layout()

                st.pyplot(fig)

                # Top collaborating countries
                degree_centrality = nx.degree_centrality(G)
                top_countries = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]

                st.subheader("🔝 Most Internationally Connected Countries")
                for i, (country, centrality) in enumerate(top_countries, 1):
                    st.write(f"{i}. **{country}** - Centrality: {centrality:.3f}")
            else:
                st.warning("No country network found with the current filter settings.")
        else:
            st.warning("Insufficient data to generate country collaboration network.")

    except ImportError:
        st.error("NetworkX and matplotlib are required. Please install them:")
        st.code("pip install networkx matplotlib")
    except Exception as e:
        st.error(f"Error generating country network: {str(e)}")

def generate_gephi_file(G, central_author):
    """Generate GEXF format file for Gephi"""
    from datetime import datetime

    # Create GEXF content
    gexf_content = '<?xml version="1.0" encoding="UTF-8"?>\n'
    gexf_content += '<gexf xmlns="http://www.gexf.net/1.2draft" version="1.2">\n'
    gexf_content += '  <meta lastmodifieddate="' + datetime.now().strftime('%Y-%m-%d') + '">\n'
    gexf_content += '    <creator>Scholar Scraper Dashboard</creator>\n'
    gexf_content += '    <description>Author collaboration network centered on ' + central_author + '</description>\n'
    gexf_content += '  </meta>\n'
    gexf_content += '  <graph mode="static" defaultedgetype="undirected">\n'

    # Add node attributes
    gexf_content += '    <attributes class="node">\n'
    gexf_content += '      <attribute id="name" title="Name" type="string"/>\n'
    gexf_content += '      <attribute id="type" title="Type" type="string"/>\n'
    gexf_content += '      <attribute id="level" title="Level" type="integer"/>\n'
    gexf_content += '      <attribute id="h_index" title="H-Index" type="float"/>\n'
    gexf_content += '    </attributes>\n'

    # Add nodes
    gexf_content += '    <nodes>\n'
    for node_id, node_data in G.nodes(data=True):
        name = node_data.get('name', str(node_id))
        node_type = node_data.get('type', 'collaborator')
        level = node_data.get('level', 0)
        h_index = node_data.get('h_index', 0)

        gexf_content += f'      <node id="{node_id}" label="{name}">\n'
        gexf_content += '        <attvalues>\n'
        gexf_content += f'          <attvalue for="name" value="{name}"/>\n'
        gexf_content += f'          <attvalue for="type" value="{node_type}"/>\n'
        gexf_content += f'          <attvalue for="level" value="{level}"/>\n'
        gexf_content += f'          <attvalue for="h_index" value="{h_index}"/>\n'
        gexf_content += '        </attvalues>\n'
        gexf_content += '      </node>\n'
    gexf_content += '    </nodes>\n'

    # Add edges
    gexf_content += '    <edges>\n'
    edge_id = 0
    for source, target in G.edges():
        gexf_content += f'      <edge id="{edge_id}" source="{source}" target="{target}"/>\n'
        edge_id += 1
    gexf_content += '    </edges>\n'

    gexf_content += '  </graph>\n'
    gexf_content += '</gexf>'

    return gexf_content

def show_downloads():
    """Display download options"""
    st.header("📥 Download Data Files")
    
    # === Profiles Data ===
    st.subheader("📊 Profiles Data")
    
    if os.path.exists(PROFILES_FILE):
        with open(PROFILES_FILE, "rb") as f:
            profiles_bytes = f.read()
        
        st.download_button(
            "📥 Download Full Profiles CSV",
            data=profiles_bytes,
            file_name="scholar_profiles.csv",
            mime="text/csv",
            help="Download the complete profiles dataset"
        )
        
        file_size = len(profiles_bytes) / (1024 * 1024)  # MB
        st.info(f"File size: {file_size:.1f} MB")
    else:
        st.warning("Profiles CSV file not found.")

    
    # === Export Options ===
    st.subheader("🔄 Export Options")
    
    df = get_profiles_df()
    if not df.empty:
        export_format = st.selectbox(
            "Choose export format",
            ["CSV", "JSON", "Excel"]
        )
        
        col1, col2 = st.columns(2)
        with col1:
            include_all = st.checkbox("Include all columns", value=True)
        with col2:
            limit_rows = st.number_input("Limit rows (0 = all)", min_value=0, value=0)
        
        if st.button("🚀 Generate Export"):
            export_df = df.copy()
            
            if not include_all:
                export_df = export_df[get_display_columns(export_df)]
            
            if limit_rows > 0:
                export_df = export_df.head(limit_rows)
            
            if export_format == "CSV":
                csv_data = export_df.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
                    csv_data,
                    file_name="exported_profiles.csv",
                    mime="text/csv"
                )
            elif export_format == "JSON":
                json_data = export_df.to_json(orient="records", indent=2)
                st.download_button(
                    "📥 Download JSON",
                    json_data,
                    file_name="exported_profiles.json",
                    mime="application/json"
                )
            elif export_format == "Excel":
                # Note: This would require openpyxl
                st.info("Excel export requires openpyxl library")

if __name__ == "__main__":
    main()
