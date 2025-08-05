import streamlit as st
import os
import re
import json
import pandas as pd
import numpy as np
from collections import Counter
from fuzzywuzzy import process
import ast

st.set_page_config(page_title="AI Researcher Database", page_icon="📚", layout="wide")

st.title("📚 Scholar Queue Enqueuer")
# Define the insert file (NOT the queue file)
INSERT_FILE = "queue_insert.jsonl"
PROFILES_FILE = "scholar_profiles_progressssss.csv"


st.markdown("Enter one or more Google Scholar **user IDs** below to enqueue (non-destructively):")

# Multi-line input
user_input = st.text_area("Scholar user IDs", placeholder="e.g. 0b_Q5gcAAAAJ, RgSc0MgAAAAJ or newline-separated")

if st.button("🚀 Enqueue Now"):
    raw_ids = user_input.replace(",", " ").replace("\n", " ").split()
    user_ids = list(set([uid.strip() for uid in raw_ids if uid.strip()]))

    if not user_ids:
        st.error("Please enter at least one valid user ID.")
    else:
        new_items = [(uid, 0, None) for uid in user_ids]

        # Prevent duplicates in insert file
        existing_ids = set()
        if os.path.exists(INSERT_FILE):
            with open(INSERT_FILE, "r") as f:
                existing_ids = {json.loads(line.strip())[0] for line in f if line.strip()}

        added = []
        skipped = []
        with open(INSERT_FILE, "a") as f:
            for item in new_items:
                if item[0] not in existing_ids:
                    f.write(json.dumps(item) + "\n")
                    added.append(item[0])
                else:
                    skipped.append(item[0])

        if added:
            st.success(f"✅ Added {len(added)} new user ID(s): {', '.join(added)}")
        if skipped:
            st.warning(f"⚠️ Skipped {len(skipped)} already enqueued ID(s): {', '.join(skipped)}")

# Award info and colors
# Award info and colors
PRESTIGIOUS_AWARDS_MATCH = [
    "nobel prize",
    "turing award",
    "fields medal",
    "rumelhart prize",
    "princess of asturias award",
    "acm a.m. turing award",
    "ieee john von neumann medal",
    "gödel prize",
    "acm prize in computing",
    "knuth prize",
    "acm grace murray hopper award",
    "c&c prize",
    "dijkstra prize",
    "nsf career award",
    "ieee computer society seymour cray computer engineering award",
    "siggraph computer graphics achievement award",
    "vinfuture prize",
    "william bowie medal",
    "wolf prize in physics",
    "wollaston medal"
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
    "ieee computer society seymour cray computer engineering award": "🏅 IEEE Seymour Cray Computer Engineering Award",
    "siggraph computer graphics achievement award": "🏅 SIGGRAPH Computer Graphics Achievement Award",
    "vinfuture prize": "🏅 VinFuture Prize",
    "william bowie medal": "🏅 William Bowie Medal",
    "wolf prize in physics": "🏅 Wolf Prize in Physics",
    "wollaston medal": "🏅 Wollaston Medal"
}

AWARD_COLORS = {
    "nobel prize": "#b8860b",                                  # dark goldenrod
    "turing award": "#1e90ff",                                 # dodger blue
    "fields medal": "#32cd32",                                 # lime green
    "rumelhart prize": "#ff69b4",                              # hot pink
    "princess of asturias award": "#8a2be2",                   # blue violet
    "acm a.m. turing award": "#1e90ff",                        # same as Turing Award - dodger blue
    "ieee john von neumann medal": "#4682b4",                  # steel blue
    "gödel prize": "#6a5acd",                                  # slate blue
    "acm prize in computing": "#20b2aa",                        # light sea green
    "knuth prize": "#ff8c00",                                  # dark orange
    "acm grace murray hopper award": "#da70d6",                # orchid
    "c&c prize": "#ff6347",                                    # tomato red
    "dijkstra prize": "#00ced1",                               # dark turquoise
    "nsf career award": "#9acd32",                             # yellow green
    "ieee computer society seymour cray computer engineering award": "#ff4500",  # orange red
    "siggraph computer graphics achievement award": "#ff1493", # deep pink
    "vinfuture prize": "#9370db",                              # medium purple
    "william bowie medal": "#8b4513",                          # saddle brown
    "wolf prize in physics": "#b8860b",                        # dark goldenrod
    "wollaston medal": "#daa520"                               # goldenrod
}

# Fellowship info and colors
PRESTIGIOUS_FELLOWSHIPS_MATCH = [
    "guggenheim fellowship",
    "macarthur fellowship",
    "sloan research fellowship",
    "packard fellowship",
    "nsf fellowship",
    "doe computational science graduate fellowship",
    "hertz fellowship",
    "stanford graduate fellowship",
    "google phd fellowship",
    "facebook fellowship",
    "microsoft research phd fellowship",
    "nvidia graduate fellowship",
    "apple scholars in aiml",
    "simons foundation fellowship",
    "simons investigator",
    "moore foundation fellowship",
    "chan zuckerberg biohub investigator",
    "chan zuckerberg initiative",
    "packard fellowships for science and engineering",
    "searle scholars program",
    "beckman young investigator",
    "arnold and mabel beckman foundation",
    "rita allen foundation scholar",
    "pew biomedical scholar",
    "james s. mcdonnell foundation",
    "templeton foundation",
    "royal society fellowship",
    "royal society research fellow",
    "leverhulme trust fellowship",
    "wellcome trust fellowship",
    "european research council",
    "erc starting grant",
    "erc consolidator grant",
    "erc advanced grant",
    "marie curie fellowship",
    "humboldt fellowship",
    "fulbright fellowship",
    "rhodes scholarship",
    "marshall scholarship",
    "churchill scholarship",
    "gates cambridge scholarship",
    "knight-hennessy scholars",
    "schwarzman scholars",
    "ieee fellow",
    "aaai fellow",
    "acm fellow",
    "wwrf fellow"
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
    "guggenheim fellowship": "#8b4513",                        # saddle brown
    "macarthur fellowship": "#800080",                         # purple
    "sloan research fellowship": "#4169e1",                    # royal blue
    "packard fellowship": "#228b22",                           # forest green
    "nsf fellowship": "#ff6347",                               # tomato
    "doe computational science graduate fellowship": "#2e8b57", # sea green
    "hertz fellowship": "#dc143c",                             # crimson
    "stanford graduate fellowship": "#8b0000",                 # dark red
    "google phd fellowship": "#4285f4",                        # google blue
    "facebook fellowship": "#1877f2",                          # facebook blue
    "microsoft research phd fellowship": "#0078d4",            # microsoft blue
    "nvidia graduate fellowship": "#76b900",                   # nvidia green
    "apple scholars in aiml": "#007aff",                       # apple blue
    "simons foundation fellowship": "#ff8c00",                 # dark orange
    "simons investigator": "#ffa500",                          # orange
    "moore foundation fellowship": "#32cd32",                  # lime green
    "chan zuckerberg biohub investigator": "#1e90ff",          # dodger blue
    "chan zuckerberg initiative": "#0080ff",                   # blue
    "packard fellowships for science and engineering": "#228b22", # forest green (same as packard)
    "searle scholars program": "#4682b4",                      # steel blue
    "beckman young investigator": "#6a5acd",                   # slate blue
    "arnold and mabel beckman foundation": "#9370db",          # medium purple
    "rita allen foundation scholar": "#da70d6",               # orchid
    "pew biomedical scholar": "#20b2aa",                       # light sea green
    "james s. mcdonnell foundation": "#b8860b",               # dark goldenrod
    "templeton foundation": "#ff69b4",                         # hot pink
    "royal society fellowship": "#8a2be2",                     # blue violet
    "royal society research fellow": "#9932cc",               # dark orchid
    "leverhulme trust fellowship": "#4b0082",                 # indigo
    "wellcome trust fellowship": "#00ced1",                   # dark turquoise
    "european research council": "#0000cd",                   # medium blue
    "erc starting grant": "#0000ff",                          # blue
    "erc consolidator grant": "#4169e1",                      # royal blue
    "erc advanced grant": "#191970",                          # midnight blue
    "marie curie fellowship": "#ff1493",                      # deep pink
    "humboldt fellowship": "#ffd700",                         # gold
    "fulbright fellowship": "#ff4500",                        # orange red
    "rhodes scholarship": "#00008b",                          # dark blue
    "marshall scholarship": "#800000",                        # maroon
    "churchill scholarship": "#2f4f4f",                       # dark slate gray
    "gates cambridge scholarship": "#008b8b",                 # dark cyan
    "knight-hennessy scholars": "#8b0000",                    # dark red
    "schwarzman scholars": "#000000",                         # black
    "ieee fellow": "#004c99",                                 # ieee blue
    "aaai fellow": "#ff6600",                                 # aaai orange
    "acm fellow": "#1e90ff",                                  # acm blue
    "wwrf fellow": "#800080"                                  # purple
}


def extract_prestigious_awards(award_str):
    if not award_str or not isinstance(award_str, str):
        return []
    chunks = [chunk.strip().lower() for chunk in re.split(r";|,| and |\n", award_str) if chunk.strip()]
    found = []
    for award in PRESTIGIOUS_AWARDS_MATCH:
        for chunk in chunks:
            if award in chunk:
                found.append(award)  # Return key (lowercase)
                break
    return found

def extract_prestigious_fellowships(fellowship_str):
    """Extract prestigious fellowships from a string similar to awards extraction"""
    if not fellowship_str or not isinstance(fellowship_str, str):
        return []
    chunks = [chunk.strip().lower() for chunk in re.split(r";|,| and |\n", fellowship_str) if chunk.strip()]
    found = []
    for fellowship in PRESTIGIOUS_FELLOWSHIPS_MATCH:
        for chunk in chunks:
            if fellowship in chunk:
                found.append(fellowship)  # Return key (lowercase)
                break
    return found

def render_colored_tags(tag_keys, colors=None):
    colors = colors or {}
    tag_html = []
    for key in tag_keys:
        # Check if it's an award or fellowship
        if key in PRESTIGIOUS_AWARDS_DISPLAY:
            display_name = PRESTIGIOUS_AWARDS_DISPLAY[key]
            color = AWARD_COLORS.get(key, "#6c757d")
        elif key in PRESTIGIOUS_FELLOWSHIPS_DISPLAY:
            display_name = PRESTIGIOUS_FELLOWSHIPS_DISPLAY[key]
            color = FELLOWSHIP_COLORS.get(key, "#6c757d")
        else:
            display_name = key.title()
            color = colors.get(key, "#6c757d")  # default gray
        
        safe_tag = display_name.replace('"', '&quot;')
        tag_html.append(
            f'<span style="background-color:{color}; color:white; padding:3px 8px; border-radius:10px; margin-right:5px; font-size:0.85em;">{safe_tag}</span>'
        )
    return " ".join(tag_html)

def get_display_columns(df):
    """Get columns for display, excluding specified columns"""
    excluded_cols = {"interest_phrases", "wiki_birth_name", "wiki_matched_title", "prestigious_awards_count", "prestigious_fellowships_count"}
    return [col for col in df.columns if col not in excluded_cols]

def extract_topic_tags(topic_str):
    """Extract topic tags from a string or list"""
    if not topic_str:
        return []
    
    if isinstance(topic_str, str):
        if topic_str.startswith('[') and topic_str.endswith(']'):
            try:
                return ast.literal_eval(topic_str)
            except:
                return topic_str.split(',')
        else:
            return [tag.strip() for tag in topic_str.split(',') if tag.strip()]
    elif isinstance(topic_str, list):
        return topic_str
    else:
        return []

st.divider()
st.header("📊 Profile Stats")
if os.path.exists(PROFILES_FILE):
    # === 📥 Load and Clean Data ===
    df = pd.read_csv(PROFILES_FILE, engine='python', on_bad_lines='skip')
    df.replace(["NaN", "nan", ""], np.nan, inplace=True)
    df.columns = [c.strip() for c in df.columns]

    # === 🧠 Compute 🇸🇬 Singapore Co-author Count ===
    required_cols = {"user_id", "country", "coauthors"}
    if required_cols.issubset(df.columns):
        id_to_country = dict(zip(df["user_id"].astype(str), df["country"].astype(str).str.lower()))

        def count_sg_coauthors(raw):
            if not isinstance(raw, str) or not raw.strip().startswith("["):
                return 0
            try:
                coauthors = ast.literal_eval(raw)
                return sum(
                    1 for c in coauthors if (
                        isinstance(c, dict) and c.get("user_id") in id_to_country and id_to_country[c["user_id"]] == "sg"
                    ) or (
                        isinstance(c, str) and c in id_to_country and id_to_country[c] == "sg"
                    )
                )
            except Exception:
                return 0

        df["num_sg_coauthors"] = df["coauthors"].apply(count_sg_coauthors)
    else:
        st.warning("⚠️ Skipping SG coauthor count: required columns not found.")

    # === 📊 Profile Statistics ===
    st.subheader("📊 Profile Stats")
    col1, col2, col3 = st.columns(3)
    col1.metric("👤 Total Profiles", len(df))

    if "h_index_all" in df:
        try:
            avg_h = round(df["h_index_all"].dropna().astype(float).mean(), 1)
            col2.metric("📈 Avg h-index", avg_h)
        except Exception:
            pass

    if "wiki_title" in df:
        wiki_count = df["wiki_title"].notna().sum()
    elif "wiki_matched_title" in df:
        wiki_count = df["wiki_matched_title"].notna().sum()
    else:
        wiki_count = None

    if wiki_count is not None:
        col3.metric("🌐 Wikipedia Matches", wiki_count)

    col4, col5, col6, col7 = st.columns(4)
    
    show_neurips = False
    show_iclr = False
    show_awards = False
    show_fellowships = False
    
    if "NeurIPS_Institution" in df:
        total_neurips = 35609
        neurips_count = df["NeurIPS_Institution"].notna().sum()
        neurips_percent = neurips_count / total_neurips * 100
        col4.metric("🧠 NeurIPS Presenters", f"{neurips_count} / {total_neurips}", help=f"{neurips_percent:.1f}%")
        if col4.button("Show NeurIPS Presenters Details"):
            show_neurips = True

    if "ICLR_Institution" in df:
        total_iclr = 27907
        iclr_count = df["ICLR_Institution"].notna().sum()
        iclr_percent = iclr_count / total_iclr * 100
        col5.metric("🧠 ICLR Presenters", f"{iclr_count} / {total_iclr}", help=f"{iclr_percent:.1f}%")
        if col5.button("Show ICLR Presenters Details"):
            show_iclr = True
    
    if "wiki_awards" in df.columns:
        df["num_awards"] = df["wiki_awards"].fillna("").apply(lambda x: len(extract_prestigious_awards(x)))
        num_award_winners = (df["num_awards"] > 0).sum()
        col6.metric("🏆 Number of Award Winners", num_award_winners)
        if col6.button("Show Award Winners Details"):
            show_awards = True

    
    # Check if we have fellowship data in awards column or separate fellowship column
    fellowship_column = None
    if "wiki_fellowships" in df.columns:
        fellowship_column = "wiki_fellowships"
    elif "wiki_awards" in df.columns:
        # Check if awards column contains fellowship info
        fellowship_column = "wiki_awards"
    
    if fellowship_column:
        df["num_fellowships"] = df[fellowship_column].fillna("").apply(lambda x: len(extract_prestigious_fellowships(x)))
        num_fellowship_holders = (df["num_fellowships"] > 0).sum()
        col7.metric("🎓 Number of Fellowship Holders", num_fellowship_holders)
        if col7.button("Show Fellowship Holders Details"):
            show_fellowships = True

    
    # Show detailed tables below if any button clicked
    display_cols = get_display_columns(df)
    
    if show_neurips:
        st.write("### 🧠 NeurIPS Presenters (out of 35609)")
        neurips_df = df[df["NeurIPS_Institution"].notna()][display_cols].reset_index(drop=True)
        st.dataframe(neurips_df, use_container_width=True)
    
    if show_iclr:
        st.write("### 🧠 ICLR Presenters (out of 27907)")
        iclr_df = df[df["ICLR_Institution"].notna()][display_cols].reset_index(drop=True)
        st.dataframe(iclr_df, use_container_width=True)
    
    if show_awards:
        st.write("### 🏆 Award Winners")
        award_winners_df = df[df["num_awards"] > 0][display_cols].sort_values("num_awards", ascending=False).reset_index(drop=True)
        st.dataframe(award_winners_df, use_container_width=True)
    
    if show_fellowships:
        st.write("### 🎓 Fellowship Holders")
        fellowship_holders_df = df[df["num_fellowships"] > 0][display_cols].sort_values("num_fellowships", ascending=False).reset_index(drop=True)
        st.dataframe(fellowship_holders_df, use_container_width=True)

    # === 🌍 Country Distribution ===
    if "country" in df:
        st.subheader("🌍 Top 10 Countries")
        top_countries = df["country"].dropna().astype(str).value_counts().head(10)
        st.bar_chart(top_countries)

    # === 🔎 Researcher Table ===
    st.subheader("🔍 Researcher Table")
    available_cols = get_display_columns(df)

    if "wiki_awards" in df.columns:
        df["prestigious_awards_count"] = df["wiki_awards"].fillna("").apply(
            lambda x: len(extract_prestigious_awards(x))
        )
        if "prestigious_awards_count" not in available_cols:
            available_cols.append("prestigious_awards_count")

    if fellowship_column:
        df["prestigious_fellowships_count"] = df[fellowship_column].fillna("").apply(
            lambda x: len(extract_prestigious_fellowships(x))
        )
        if "prestigious_fellowships_count" not in available_cols:
            available_cols.append("prestigious_fellowships_count")

    st.dataframe(df[available_cols].dropna(how="all"), use_container_width=True)

    # === 🏅 Enhanced Filter by Category ===
    st.divider()
    st.header("🔍 Filter Authors by Category")

    # First dropdown for broad category selection
    filter_category = st.selectbox(
        "🎯 Select Filter Category",
        options=["None", "Awards", "Fellowships", "Institutions", "Country", "Topic Tags", "Prestigious Awards", "Prestigious Fellowships"]
    )

    filtered_df = df.copy()
    
    if filter_category != "None":
        if filter_category == "Awards":
            if "wiki_awards" in df.columns:
                # Get all unique awards
                all_awards = set()
                for awards_str in df["wiki_awards"].dropna():
                    if isinstance(awards_str, str) and awards_str.strip():
                        awards = [award.strip() for award in re.split(r';|,|\n', awards_str) if award.strip()]
                        all_awards.update(awards)
                
                award_options = ["Any Award"] + sorted(list(all_awards))
                selected_award = st.selectbox("🏆 Select Specific Award", award_options)
                
                if selected_award != "Any Award":
                    filtered_df = df[df["wiki_awards"].fillna("").str.contains(selected_award, case=False, na=False)]
                else:
                    filtered_df = df[df["wiki_awards"].notna() & (df["wiki_awards"].str.strip() != "")]
            else:
                st.warning("Awards column not found in data.")
        
        elif filter_category == "Fellowships":
            if fellowship_column:
                # Get all unique fellowships
                all_fellowships = set()
                for fellowship_str in df[fellowship_column].dropna():
                    if isinstance(fellowship_str, str) and fellowship_str.strip():
                        fellowships = [fellowship.strip() for fellowship in re.split(r';|,|\n', fellowship_str) if fellowship.strip()]
                        all_fellowships.update(fellowships)
                
                fellowship_options = ["Any Fellowship"] + sorted(list(all_fellowships))
                selected_fellowship = st.selectbox("🎓 Select Specific Fellowship", fellowship_options)
                
                if selected_fellowship != "Any Fellowship":
                    filtered_df = df[df[fellowship_column].fillna("").str.contains(selected_fellowship, case=False, na=False)]
                else:
                    filtered_df = df[df[fellowship_column].notna() & (df[fellowship_column].str.strip() != "")]
            else:
                st.warning("Fellowship column not found in data.")
                
        elif filter_category == "Institutions":
            if "institution" in df.columns:
                institutions = sorted(df["institution"].dropna().unique())
                selected_institution = st.selectbox("🏛️ Select Institution", ["All"] + institutions)
                
                if selected_institution != "All":
                    filtered_df = df[df["institution"] == selected_institution]
            else:
                st.warning("Institution column not found in data.")
                
        elif filter_category == "Country":
            if "country" in df.columns:
                countries = sorted(df["country"].dropna().unique())
                selected_countries = st.multiselect("🌍 Select Countries", countries, default=countries)
                filtered_df = df[df["country"].isin(selected_countries)]
            else:
                st.warning("Country column not found in data.")
                
        elif filter_category == "Topic Tags":
            if "topic_tags" in df.columns:
                # Extract all unique topic tags from the topic_tags column
                all_topics = set()
                for topic_data in df["topic_tags"].dropna():
                    topics = extract_topic_tags(topic_data)
                    all_topics.update([str(topic).strip() for topic in topics if str(topic).strip()])
                
                if all_topics:
                    # Allow multiple selection of tags
                    selected_topics = st.multiselect(
                        "🏷️ Select Topic Tags", 
                        sorted(list(all_topics)),
                        help="Select one or more topic tags to filter by (ALL selected tags must be present)"
                    )
                    
                    if selected_topics:
                        # Filter rows that contain ALL of the selected topics
                        def contains_all_topics(topic_list):
                            if not topic_list:
                                return False
                            topics = extract_topic_tags(topic_list)
                            # Check if ALL selected topics are present in the author's topics
                            return all(topic in topics for topic in selected_topics)
                        
                        mask = df["topic_tags"].apply(contains_all_topics)
                        filtered_df = df[mask]
                    else:
                        # If no topics selected, show all rows with topic tags
                        filtered_df = df[df["topic_tags"].notna()]
                else:
                    st.info("No topic tags found in the data.")
            else:
                st.warning("topic_tags column not found in data.")
                
        elif filter_category == "Prestigious Awards":
            award_choice = st.selectbox(
                "🎖️ Select Prestigious Award",
                options=["All Prestigious Awards"] + list(PRESTIGIOUS_AWARDS_DISPLAY.values()) + ["No Prestigious Awards"]
            )

            if award_choice == "All Prestigious Awards":
                # Show only those with any prestigious award
                filtered_df = df[df["wiki_awards"].fillna("").apply(lambda x: len(extract_prestigious_awards(x)) > 0)]
            elif award_choice == "No Prestigious Awards":
                # Show only those without prestigious awards
                filtered_df = df[df["wiki_awards"].fillna("").apply(lambda x: len(extract_prestigious_awards(x)) == 0)]
            elif award_choice in PRESTIGIOUS_AWARDS_DISPLAY.values():
                key = next(k for k, v in PRESTIGIOUS_AWARDS_DISPLAY.items() if v == award_choice)
                filtered_df = df[df["wiki_awards"].fillna("").str.lower().str.contains(key)]
        
        elif filter_category == "Prestigious Fellowships":
            if fellowship_column:
                fellowship_choice = st.selectbox(
                    "🎓 Select Prestigious Fellowship",
                    options=["All Prestigious Fellowships"] + list(PRESTIGIOUS_FELLOWSHIPS_DISPLAY.values()) + ["No Prestigious Fellowships"]
                )

                if fellowship_choice == "All Prestigious Fellowships":
                    # Show only those with any prestigious fellowship
                    filtered_df = df[df[fellowship_column].fillna("").apply(lambda x: len(extract_prestigious_fellowships(x)) > 0)]
                elif fellowship_choice == "No Prestigious Fellowships":
                    # Show only those without prestigious fellowships
                    filtered_df = df[df[fellowship_column].fillna("").apply(lambda x: len(extract_prestigious_fellowships(x)) == 0)]
                elif fellowship_choice in PRESTIGIOUS_FELLOWSHIPS_DISPLAY.values():
                    key = next(k for k, v in PRESTIGIOUS_FELLOWSHIPS_DISPLAY.items() if v == fellowship_choice)
                    filtered_df = df[df[fellowship_column].fillna("").str.lower().str.contains(key)]
            else:
                st.warning("Fellowship column not found in data.")

    # Display filtered results
    if filtered_df.empty:
        st.info(f"🔎 No authors found for the selected filter: **{filter_category}**")
    else:
        st.write(f"📊 Showing {len(filtered_df)} author(s) after filtering by {filter_category}")
        display_cols = get_display_columns(filtered_df)
        st.dataframe(filtered_df[display_cols], use_container_width=True)

else:
    st.info("📂 No profiles file found yet. Start crawling to generate one.")


st.divider()
st.header("🔍 Search Authors by Name (Fuzzy)")

search_query = st.text_input("Enter author name to search")

def fuzzy_author_search(df, query, score_cutoff=70):
    if df.empty or not query.strip():
        return pd.DataFrame()
    names = df["name"].dropna().unique()
    results = process.extract(query, names, limit=None)
    filtered = [r for r in results if r[1] >= score_cutoff]
    matched_names = [r[0] for r in filtered]
    if not matched_names:
        return pd.DataFrame()
    return df[df["name"].isin(matched_names)].copy()

def fuzzy_institution_search(df, query, score_cutoff=70):
    if df.empty or not query.strip() or "institution" not in df.columns:
        return pd.DataFrame()
    institutions = df["institution"].dropna().unique()
    results = process.extract(query, institutions, limit=None)
    filtered = [r for r in results if r[1] >= score_cutoff]
    matched_insts = [r[0] for r in filtered]
    if not matched_insts:
        return pd.DataFrame()
    return df[df["institution"].isin(matched_insts)].copy()

def display_author_info(author_row):
    name = author_row.get("name", "Unknown")
    participated_iclr = author_row.get("Participated_in_ICLR", False)
    participated_neurips = author_row.get("Participated_in_NeurIPS", False)

    # === Extract awards and fellowships ===
    awards_str = author_row.get("wiki_awards", "")
    fellowship_str = author_row.get("wiki_fellowships", "") or awards_str  # fallback to awards if no separate fellowship column
    
    prestigious_awards = extract_prestigious_awards(awards_str)
    prestigious_fellowships = extract_prestigious_fellowships(fellowship_str)
    
    # Combine awards and fellowships for display
    all_honors = prestigious_awards + prestigious_fellowships
    honors_tags_md = ""
    if all_honors:
        honors_tags_md = render_colored_tags(all_honors).replace("\n", " ")

    # === Line 1: Name + Awards/Fellowships ===
    st.markdown(
        f"<div style='font-size: 1.4em; font-weight: bold;'>{name} {honors_tags_md}</div>",
        unsafe_allow_html=True,
    )

    # === Line 2: Conference Participation Badges ===
    badges = []
    if participated_iclr in [True, "TRUE", "True"]:
        badges.append(
            "<span style='color: white; background-color: #4CAF50; "
            "padding: 3px 8px; border-radius: 5px; font-size: 1em;'>🏆 ICLR</span>"
        )
    if participated_neurips in [True, "TRUE", "True"]:
        badges.append(
            "<span style='color: white; background-color: #0072C6; "
            "padding: 3px 8px; border-radius: 5px; font-size: 1em;'>🎖️ NeurIPS</span>"
        )
    if badges:
        st.markdown(" ".join(badges), unsafe_allow_html=True)

    # === Show other fields (excluding the ones we don't want to display) ===
    excluded_fields = {"name", "interest_phrases", "wiki_birth_name", "wiki_matched_title", "num_awards", "num_fellowships"}
    for col, val in author_row.items():
        if col in excluded_fields:
            continue
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        if isinstance(val, str) and val.strip() == "":
            continue
        if isinstance(val, list) and len(val) == 0:
            continue
        if isinstance(val, list):
            val = ", ".join(str(x) for x in val)
        elif isinstance(val, str):
            val = val.strip()
        label = col.replace("_", " ").title()
        st.markdown(f"**{label}:** {val}")


if search_query:
    if os.path.exists(PROFILES_FILE):
        df = pd.read_csv(PROFILES_FILE, engine='python', on_bad_lines='skip')

        # Search by name
        matches_df = fuzzy_author_search(df, search_query, score_cutoff=70)

        # If no matches by name, try institution search
        if matches_df.empty:
            matches_df = fuzzy_institution_search(df, search_query, score_cutoff=70)

        if matches_df.empty:
            st.info("No authors found matching your query.")
        else:
            # === Filters ===
            with st.expander("🔧 Filter & Sort Options", expanded=True):
                # Filter by country
                if "country" in matches_df.columns:
                    countries = sorted(matches_df["country"].dropna().unique())
                    selected_countries = st.multiselect("🌍 Filter by Country", countries, default=countries)
                    matches_df = matches_df[matches_df["country"].isin(selected_countries)]

                # Filter by prestigious awards
                if "wiki_awards" in matches_df.columns:
                    award_filter = st.selectbox(
                        "🏅 Filter by Prestigious Award",
                        options=["Any"] + list(PRESTIGIOUS_AWARDS_DISPLAY.values()) + ["None"],
                        index=0
                    )
                    if award_filter == "None":
                        matches_df = matches_df[matches_df["wiki_awards"].fillna("").str.strip() == ""]
                    elif award_filter != "Any":
                        award_key = next(k for k, v in PRESTIGIOUS_AWARDS_DISPLAY.items() if v == award_filter)
                        matches_df = matches_df[matches_df["wiki_awards"].fillna("").str.lower().str.contains(award_key)]

                # Filter by prestigious fellowships
                fellowship_column = "wiki_fellowships" if "wiki_fellowships" in matches_df.columns else "wiki_awards"
                if fellowship_column in matches_df.columns:
                    fellowship_filter = st.selectbox(
                        "🎓 Filter by Prestigious Fellowship",
                        options=["Any"] + list(PRESTIGIOUS_FELLOWSHIPS_DISPLAY.values()) + ["None"],
                        index=0
                    )
                    if fellowship_filter == "None":
                        matches_df = matches_df[matches_df[fellowship_column].fillna("").apply(lambda x: len(extract_prestigious_fellowships(x)) == 0)]
                    elif fellowship_filter != "Any":
                        fellowship_key = next(k for k, v in PRESTIGIOUS_FELLOWSHIPS_DISPLAY.items() if v == fellowship_filter)
                        matches_df = matches_df[matches_df[fellowship_column].fillna("").str.lower().str.contains(fellowship_key)]

                # Filter by h-index
                if "h_index_all" in matches_df.columns:
                    min_h = int(np.nanmin(matches_df["h_index_all"]))
                    max_h = int(np.nanmax(matches_df["h_index_all"]))

                    if min_h == max_h:
                        st.info(f"All matching authors have the same h-index: {min_h}")
                        h_range = (min_h, max_h)
                    else:
                        h_range = st.slider(
                            "📈 Filter by h-index",
                            min_value=min_h,
                            max_value=max_h,
                            value=(min_h, max_h)
                        )

                    matches_df = matches_df[
                        (matches_df["h_index_all"].astype(float) >= h_range[0]) &
                        (matches_df["h_index_all"].astype(float) <= h_range[1])
                    ]

                # Sort
                sort_by = st.selectbox("📊 Sort by", ["Name", "h-index", "Country"], index=0)
                sort_order = st.radio("🔃 Order", ["Ascending", "Descending"], horizontal=True, index=0)

                sort_map = {
                    "Name": "name",
                    "h-index": "h_index_all",
                    "Country": "country"
                }
                sort_col = sort_map[sort_by]
                matches_df = matches_df.sort_values(
                    by=sort_col,
                    ascending=(sort_order == "Ascending"),
                    na_position="last"
                )

            # === Display Results ===
            if matches_df.empty:
                st.info("No authors found after applying filters.")
            else:
                st.write(f"Showing {len(matches_df)} author(s) matching `{search_query}`:")
                for idx, row in matches_df.iterrows():
                    with st.expander(row["name"]):
                        display_author_info(row)
                
                # Use display columns for CSV download
                display_cols = get_display_columns(matches_df)
                matches_csv = matches_df[display_cols].to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Filtered Search Results CSV",
                    data=matches_csv,
                    file_name="author_search_filtered_results.csv",
                    mime="text/csv"
                )
    else:
        st.info("Profiles file not found yet.")

else:
    st.info("Enter a search query above to find authors.")

# === New feature: Bulk upload user IDs from CSV file ===
st.divider()
st.header("📤 Bulk Upload User IDs from CSV File")

uploaded_file = st.file_uploader("Upload CSV file with a 'user_id' column", type=["csv"])

if uploaded_file is not None:
    try:
        upload_df = pd.read_csv(uploaded_file)
        if "user_id" not in upload_df.columns:
            st.error("CSV must contain a 'user_id' column.")
        else:
            uploaded_ids = upload_df["user_id"].dropna().astype(str).str.strip().unique().tolist()
            st.write(f"Found {len(uploaded_ids)} unique user IDs in uploaded file.")

            if st.button("➕ Enqueue Uploaded IDs"):
                # Prevent duplicates in insert file
                existing_ids = set()
                if os.path.exists(INSERT_FILE):
                    with open(INSERT_FILE, "r") as f:
                        existing_ids = {json.loads(line.strip())[0] for line in f if line.strip()}

                added = []
                skipped = []
                with open(INSERT_FILE, "a") as f:
                    for uid in uploaded_ids:
                        if uid and uid not in existing_ids:
                            f.write(json.dumps((uid, 0, None)) + "\n")
                            added.append(uid)
                        else:
                            skipped.append(uid)
                if added:
                    st.success(f"✅ Added {len(added)} new user ID(s): {', '.join(added)}")
                if skipped:
                    st.warning(f"⚠️ Skipped {len(skipped)} already enqueued ID(s): {', '.join(skipped)}")

    except Exception as e:
        st.error(f"Error reading uploaded file: {e}")

# === New feature: Download entire Profiles CSV & Insert Queue ===
st.divider()
st.header("📥 Download Data Files")

if os.path.exists(PROFILES_FILE):
    with open(PROFILES_FILE, "rb") as f:
        profiles_bytes = f.read()
    st.download_button("Download Full Profiles CSV", data=profiles_bytes, file_name="full_profiles.csv", mime="text/csv")
else:
    st.info("Profiles CSV file not found for download.")

if os.path.exists(INSERT_FILE):
    with open(INSERT_FILE, "r") as f:
        insert_content = f.read()
    st.download_button("Download Full Insert Queue File", data=insert_content, file_name="queue_insert.jsonl", mime="text/plain")
else:
    st.info("Insert queue file not found for download.")

st.divider()
st.header("📂 Current Queue Insert File")

if os.path.exists(INSERT_FILE):
    with open(INSERT_FILE, "r") as f:
        lines = f.readlines()
    
    if lines:
        st.write(f"📋 Currently {len(lines)} items in queue:")
        queue_data = []
        for line in lines[-10:]:  # Show last 10 items
            try:
                item = json.loads(line.strip())
                queue_data.append({"User ID": item[0], "Priority": item[1], "Status": item[2] or "Pending"})
            except:
                continue
        
        if queue_data:
            queue_df = pd.DataFrame(queue_data)
            st.dataframe(queue_df, use_container_width=True)
            
            if len(lines) > 10:
                st.info(f"Showing last 10 items. Total items in queue: {len(lines)}")
    else:
        st.info("Queue is currently empty.")
else:
    st.info("No insert file found yet. Add users to begin populating it.")
