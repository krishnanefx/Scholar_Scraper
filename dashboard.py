import streamlit as st
import os
import re
import json
import pandas as pd
import numpy as np
from collections import Counter
from fuzzywuzzy import process

st.set_page_config(page_title="Scholar Queue Enqueuer", page_icon="📚", layout="wide")

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
    "ieee computer society seymour cray computer engineering award"
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
    "ieee computer society seymour cray computer engineering award": "🏅 IEEE Seymour Cray Computer Engineering Award"
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
    "ieee computer society seymour cray computer engineering award": "#ff4500"  # orange red
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

def render_colored_tags(tag_keys, colors=None):
    colors = colors or {}
    tag_html = []
    for key in tag_keys:
        display_name = PRESTIGIOUS_AWARDS_DISPLAY.get(key, key.title())
        color = colors.get(key, "#6c757d")  # default gray
        safe_tag = display_name.replace('"', '&quot;')
        tag_html.append(
            f'<span style="background-color:{color}; color:white; padding:3px 8px; border-radius:10px; margin-right:5px; font-size:0.85em;">{safe_tag}</span>'
        )
    return " ".join(tag_html)

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
        import ast
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

    col4, col5, col6 = st.columns(3)
    
    show_neurips = False
    show_iclr = False
    show_awards = False
    
    if "NeurIPS_Institution" in df:
        neurips_count = df["NeurIPS_Institution"].notna().sum()
        col4.metric("🧠 NeurIPS Presenters", neurips_count)
        if col4.button("Show NeurIPS Presenters Details"):
            show_neurips = True
    
    if "ICLR_Institution" in df:
        iclr_count = df["ICLR_Institution"].notna().sum()
        col5.metric("🧠 ICLR Presenters", iclr_count)
        if col5.button("Show ICLR Presenters Details"):
            show_iclr = True
    
    if "wiki_awards" in df.columns:
        df["num_awards"] = df["wiki_awards"].fillna("").apply(lambda x: len(extract_prestigious_awards(x)))
        num_award_winners = (df["num_awards"] > 0).sum()
        col6.metric("🏆 Number of Award Winners", num_award_winners)
        if col6.button("Show Award Winners Details"):
            show_awards = True
    
    # Show detailed tables below if any button clicked
    if show_neurips:
        st.write("### 🧠 NeurIPS Presenters")
        neurips_df = df[df["NeurIPS_Institution"].notna()][
            ["name", "institution", "country"]
        ].reset_index(drop=True)
        st.dataframe(neurips_df, use_container_width=True)
    
    if show_iclr:
        st.write("### 🧠 ICLR Presenters")
        iclr_df = df[df["ICLR_Institution"].notna()][
            ["name", "institution", "country"]
        ].reset_index(drop=True)
        st.dataframe(iclr_df, use_container_width=True)
    
    if show_awards:
        st.write("### 🏆 Award Winners")
        award_winners_df = df[df["num_awards"] > 0][
            ["name", "institution", "country", "num_awards"]
        ].sort_values("num_awards", ascending=False).reset_index(drop=True)
        st.dataframe(award_winners_df, use_container_width=True)

    # === 🌍 Country Distribution ===
    if "country" in df:
        st.subheader("🌍 Top 10 Countries")
        top_countries = df["country"].dropna().astype(str).value_counts().head(10)
        st.bar_chart(top_countries)

    # === 🔎 Researcher Table ===
    st.subheader("🔍 Researcher Table")
    available_cols = df.columns.tolist()

    if "wiki_awards" in df.columns:
        df["prestigious_awards_count"] = df["wiki_awards"].fillna("").apply(
            lambda x: len(extract_prestigious_awards(x))
        )
        available_cols.append("prestigious_awards_count")

    st.dataframe(df[available_cols].dropna(how="all"), use_container_width=True)

    # === 🏅 Filter by Prestigious Awards ===
    st.divider()
    st.header("🏅 Filter Authors by Prestigious Award")

    award_choice = st.selectbox(
        "🎖️ Select Award to Filter",
        options=["All"] + list(PRESTIGIOUS_AWARDS_DISPLAY.values())
    )

    if award_choice == "All":
        filtered_df = df
    else:
        key = next(k for k, v in PRESTIGIOUS_AWARDS_DISPLAY.items() if v == award_choice)
        filtered_df = df[df["wiki_awards"].fillna("").str.lower().str.contains(key)]

    if filtered_df.empty:
        st.info(f"🔎 No authors found for award: **{award_choice}**")
    else:
        display_df = filtered_df.copy()
        display_df.columns = [col.replace("_", " ").title() for col in display_df.columns]
        st.dataframe(display_df, use_container_width=True)

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

    # === Extract awards ===
    awards_str = author_row.get("wiki_awards", "")
    prestigious_awards = extract_prestigious_awards(awards_str)
    awards_tags_md = ""
    if prestigious_awards:
        awards_tags_md = render_colored_tags(prestigious_awards, AWARD_COLORS).replace("\n", " ")

    # === Line 1: Name + Awards ===
    st.markdown(
        f"<div style='font-size: 1.4em; font-weight: bold;'>{name} {awards_tags_md}</div>",
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

    # === Show other fields ===
    for col, val in author_row.items():
        if col in ["name"]:
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
            st.write(f"Showing {len(matches_df)} author(s) matching `{search_query}`:")
            for idx, row in matches_df.iterrows():
                with st.expander(row["name"]):
                    display_author_info(row)
            matches_csv = matches_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Search Results CSV",
                data=matches_csv,
                file_name="author_search_results.csv",
                mime="text/csv"
            )
    else:
        st.info("Profiles file not found yet.")

st.divider()
st.header("📂 Current Queue Insert File")

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
                matches_csv = matches_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Filtered Search Results CSV",
                    data=matches_csv,
                    file_name="author_search_filtered_results.csv",
                    mime="text/csv"
                )
    else:
        st.info("Profiles file not found yet.")

else:
    st.info("No insert file found yet. Add users to begin populating it.")

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