#!/usr/bin/env python3
"""
2026 Conference Predictions with Scholar Profiles
Creates Excel file with scholar profiles for AAAI, NeurIPS, and ICLR predicted participants
Matches predicted authors with scholar profile database using fuzzy matching
"""

import argparse
import datetime
import os
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz, process
import re
from tqdm import tqdm
import joblib

def clean_name(name):
    """Clean and normalize author names for better matching"""
    if pd.isna(name):
        return ""
    
    # Convert to string and lowercase
    name = str(name).lower()
    
    # Remove common prefixes/suffixes
    prefixes = ['prof.', 'professor', 'dr.', 'mr.', 'ms.', 'mrs.']
    for prefix in prefixes:
        if name.startswith(prefix):
            name = name[len(prefix):].strip()
    
    # Remove special characters and extra spaces
    name = re.sub(r'[^\w\s]', ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name

def match_authors_with_profiles(predicted_participants, scholar_profiles, conference_name, threshold=80):
    """
    Match predicted participants with scholar profiles using fuzzy string matching
    """
    print(f"Matching {conference_name} predicted participants with scholar profiles...")
    
    # Clean names for better matching
    predicted_participants['clean_name'] = predicted_participants['predicted_author'].apply(clean_name)
    scholar_profiles['clean_name'] = scholar_profiles['name'].apply(clean_name)
    
    # Create lookup dictionary for scholar profiles
    scholar_lookup = {}
    for idx, row in scholar_profiles.iterrows():
        clean_scholar_name = row['clean_name']
        if clean_scholar_name and clean_scholar_name.strip():
            if clean_scholar_name not in scholar_lookup:
                scholar_lookup[clean_scholar_name] = []
            scholar_lookup[clean_scholar_name].append({
                'index': idx,
                'original_name': row['name'],
                'data': row
            })
    
    # Prepare results
    matched_results = []
    unmatched_authors = []

    def _get_pred_value(series, base_name, default=''):
        """Return the first matching column value from a prediction series for a given base name."""
        # exact match
        if base_name in series.index:
            return series[base_name]
        # look for contains or startswith
        for col in series.index:
            if base_name in str(col) or str(col).startswith(base_name):
                return series[col]
        return default
    
    for idx, predicted_row in tqdm(predicted_participants.iterrows(),
                                  total=len(predicted_participants),
                                  desc=f"Matching {conference_name}"):
        predicted_name = predicted_row.get('clean_name', '')
        original_predicted_name = predicted_row.get('predicted_author', predicted_row.get('author', ''))
        
        best_match = None
        best_score = 0
        
        # Try exact match first
        if predicted_name in scholar_lookup:
            best_match = scholar_lookup[predicted_name][0]
            best_score = 100
        else:
            # Use fuzzy matching
            scholar_names = list(scholar_lookup.keys())
            if scholar_names:
                match_result = process.extractOne(predicted_name, scholar_names, scorer=fuzz.ratio)
                if match_result and match_result[1] >= threshold:
                    matched_scholar_name = match_result[0]
                    best_match = scholar_lookup[matched_scholar_name][0]
                    best_score = match_result[1]

        if best_match and best_score >= threshold:
            # Combine prediction data with scholar profile data
            result_row = {
                # Conference info
                'Conference': conference_name,

                # Prediction data (use flexible column matching)
                'Predicted_Author': original_predicted_name,
                'Will_Participate': _get_pred_value(predicted_row, 'will_participate'),
                'Participation_Probability': _get_pred_value(predicted_row, 'participation_probability'),
                'Confidence_Percent': _get_pred_value(predicted_row, 'confidence_percent'),
                'Prediction_Rank': _get_pred_value(predicted_row, 'rank'),
                'Past_Participations': _get_pred_value(predicted_row, 'num_participations'),
                'Years_Since_Last': _get_pred_value(predicted_row, 'years_since_last'),
                'Participation_Rate': _get_pred_value(predicted_row, 'participation_rate'),

                # Matching info
                'Match_Score': best_score,
                'Scholar_Profile_Name': best_match['original_name'],

                # Scholar profile data
                'User_ID': best_match['data'].get('user_id', ''),
                'Position': best_match['data'].get('position', ''),
                'Email': best_match['data'].get('email', ''),
                'Homepage': best_match['data'].get('homepage', ''),
                'Country': best_match['data'].get('country', ''),
                'Institution': best_match['data'].get('institution', ''),
                'Research_Interests': best_match['data'].get('research_interests', ''),
                'Interest_Phrases': best_match['data'].get('interest_phrases', ''),
                'Citations_All': best_match['data'].get('citations_all', ''),
                'H_Index_All': best_match['data'].get('h_index_all', ''),
                'Topic_Tags': best_match['data'].get('topic_tags', ''),
                'Coauthors': best_match['data'].get('coauthors', ''),

                # Conference participation history
                'Participated_in_ICLR': best_match['data'].get('Participated_in_ICLR', ''),
                'ICLR_Institution': best_match['data'].get('ICLR_Institution', ''),
                'Participated_in_NeurIPS': best_match['data'].get('Participated_in_NeurIPS', ''),
                'NeurIPS_Institution': best_match['data'].get('NeurIPS_Institution', ''),

                # Wikipedia data
                'Wiki_Name': best_match['data'].get('wiki_name', ''),
                'Wiki_Birth_Date': best_match['data'].get('wiki_birth_date', ''),
                'Wiki_Birth_Place': best_match['data'].get('wiki_birth_place', ''),
                'Wiki_Fields': best_match['data'].get('wiki_fields', ''),
                'Wiki_Work_Institution': best_match['data'].get('wiki_work_institution', ''),
                'Wiki_Alma_Mater': best_match['data'].get('wiki_alma_mater', ''),
                'Wiki_Known_For': best_match['data'].get('wiki_known_for', ''),
                'Wiki_Awards': best_match['data'].get('wiki_awards', ''),
                'Wiki_Summary': best_match['data'].get('wiki_wiki_summary', ''),
                'Wiki_Is_Researcher_ML': best_match['data'].get('wiki_is_researcher_ml', '')
            }
            # Attach all prediction columns (prefix with Pred_ to avoid collisions)
            try:
                pred_dict = predicted_row.to_dict()
                for k, v in pred_dict.items():
                    # skip fields we already added
                    if k in ['predicted_author', 'clean_name']:
                        continue
                    result_row[f'Pred_{k}'] = v
            except Exception:
                pass

            matched_results.append(result_row)
        else:
            unmatched_authors.append({
                'Conference': conference_name,
                'Predicted_Author': original_predicted_name,
                'Participation_Probability': _get_pred_value(predicted_row, 'participation_probability'),
                'Prediction_Rank': _get_pred_value(predicted_row, 'rank')
            })
    
    return matched_results, unmatched_authors

def main():
    parser = argparse.ArgumentParser(description='Match conference prediction CSVs with scholar profiles and export Excel reports')
    parser.add_argument('--year', '-y', type=int, default=datetime.datetime.now().year, help='Prediction year to process (default: current year)')
    parser.add_argument('--threshold', '-t', type=int, default=80, help='Fuzzy matching threshold (0-100)')
    args = parser.parse_args()

    year = args.year
    threshold = args.threshold

    print(f"=== {year} Conference Predictions with Scholar Profile Matching ===\n")
    
    # Load scholar profiles database
    # Resolve project root relative to this script, so script can be run from repo root or other cwd
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))

    print("Loading scholar profiles database...")
    try:
        scholar_profiles_path = os.path.join(project_root, 'data', 'processed', 'scholar_profiles.csv')
        scholar_profiles = pd.read_csv(scholar_profiles_path, low_memory=False)
        print(f"✅ Loaded {len(scholar_profiles)} scholar profiles (from {scholar_profiles_path})")
    except Exception as e:
        print(f"❌ Error loading scholar profiles: {e}")
        return
    
    # Load all prediction files dynamically for common conferences
    conferences = ['aaai', 'neurips', 'iclr']
    conferences_data = {}

    for conf in conferences:
        pred_path = os.path.join(project_root, 'data', 'predictions', f'{conf}_{year}_predictions.csv')
        try:
            preds = pd.read_csv(pred_path)
            # Determine participant filter column flexibly
            will_col = None
            for c in preds.columns:
                if 'will_participate' in c:
                    will_col = c
                    break
            if will_col:
                participants = preds[preds[will_col] == 1]
            else:
                # fallback: use a 'predicted_author' presence
                participants = preds[~preds['predicted_author'].isna()] if 'predicted_author' in preds.columns else preds

            conferences_data[conf.upper()] = participants
            print(f"✅ {conf.upper()} {year} predictions loaded: {len(participants)} predicted participants (source: {pred_path})")
        except Exception as e:
            print(f"❌ {conf.upper()} predictions not found at {pred_path}: {e}")
            conferences_data[conf.upper()] = pd.DataFrame()
    
    # Perform matching for each conference
    all_matched_results = []
    all_unmatched_results = []
    conference_results = {}
    
    print(f"\n📋 Starting fuzzy matching process...")
    
    for conf_name, participants_df in conferences_data.items():
        if not participants_df.empty:
            matched_results, unmatched_results = match_authors_with_profiles(
                participants_df, scholar_profiles, conf_name, threshold=80
            )
            
            if matched_results:
                conference_results[conf_name] = pd.DataFrame(matched_results)
                all_matched_results.extend(matched_results)
                print(f"   • {conf_name}: {len(matched_results)} matched, {len(unmatched_results)} unmatched")
            else:
                conference_results[conf_name] = pd.DataFrame()
                print(f"   • {conf_name}: No matches found")
            
            all_unmatched_results.extend(unmatched_results)
    
    # Create Excel file with results
    outputs_dir = os.path.join(project_root, 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    excel_filename = os.path.join(outputs_dir, f'{year}_Conference_Predictions_with_Scholar_Profiles.xlsx')
    
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        
        # Create individual conference sheets
        model_summaries = []

        for conf_name, results_df in conference_results.items():
            if not results_df.empty:
                # Sort by prediction rank
                results_df = results_df.sort_values('Prediction_Rank')
                
                # Write to Excel sheet
                results_df.to_excel(writer, sheet_name=f'{conf_name}_2026', index=False)
                
                print(f"\n📊 {conf_name} 2026 Sheet Created:")
                print(f"   • Total matched participants: {len(results_df)}")
                print(f"   • Average H-index: {pd.to_numeric(results_df['H_Index_All'], errors='coerce').mean():.1f}")
                print(f"   • Average Citations: {pd.to_numeric(results_df['Citations_All'], errors='coerce').mean():.0f}")
                
                # Show top institutions
                if 'Institution' in results_df.columns:
                    top_institutions = results_df['Institution'].value_counts().head(5)
                    print(f"   • Top institutions: {', '.join(top_institutions.index[:3])}")

                # Attempt to load model metadata from processed folder (e.g., iclr_participation_model.pkl)
                    model_path = os.path.join(project_root, 'data', 'processed', f'{conf_name.lower()}_participation_model.pkl')
                try:
                    model = joblib.load(model_path)
                    summary = {'Conference': conf_name, 'Model_Path': model_path}
                    # sklearn-like estimators
                    if hasattr(model, 'feature_importances_'):
                        summary['Feature_Importances'] = list(model.feature_importances_)
                        summary['Feature_Names'] = list(getattr(model, 'feature_names_in_', []))
                    elif hasattr(model, 'coef_'):
                        summary['Coefficients'] = list(np.ravel(model.coef_))
                        summary['Feature_Names'] = list(getattr(model, 'feature_names_in_', []))
                    else:
                        summary['Info'] = str(type(model))
                    model_summaries.append(summary)
                except Exception:
                    # ignore if model not available
                    pass
        
    # Create summary sheet
        summary_data = []
        
        for conf_name, results_df in conference_results.items():
            if not results_df.empty:
                total_predicted = len(conferences_data[conf_name])
                matched_count = len(results_df)
                match_rate = (matched_count / total_predicted * 100) if total_predicted > 0 else 0
                
                avg_prob = results_df['Participation_Probability'].mean()
                avg_h_index = pd.to_numeric(results_df['H_Index_All'], errors='coerce').mean()
                avg_citations = pd.to_numeric(results_df['Citations_All'], errors='coerce').mean()
                
                summary_data.append({
                    'Conference': f'{conf_name} {year}',
                    'Total_Predictions': total_predicted,
                    'Matched_Profiles': matched_count,
                    'Match_Rate': f"{match_rate:.1f}%",
                    'Avg_Probability': f"{avg_prob:.3f}",
                    'Avg_H_Index': f"{avg_h_index:.1f}",
                    'Avg_Citations': f"{avg_citations:.0f}",
                    'Top_Institution': results_df['Institution'].value_counts().index[0] if not results_df['Institution'].value_counts().empty else 'N/A'
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            print(f"\n📈 Summary sheet created")

        # Write raw predictions (for traceability) into separate sheets
        for conf_key, preds_df in conferences_data.items():
            if not preds_df.empty:
                try:
                    preds_df.to_excel(writer, sheet_name=f'Raw_{conf_key}_{year}', index=False)
                except Exception:
                    # sheet name might be too long or invalid; skip silently
                    pass

        # Write model summaries if any
        if model_summaries:
            ms_df = pd.DataFrame(model_summaries)
            try:
                ms_df.to_excel(writer, sheet_name='Model_Summaries', index=False)
            except Exception:
                pass
        
        # Create unmatched authors sheet for manual review
        if all_unmatched_results:
            unmatched_df = pd.DataFrame(all_unmatched_results)
            unmatched_df = unmatched_df.sort_values(['Conference', 'Prediction_Rank'])
            unmatched_df.to_excel(writer, sheet_name='Unmatched_Authors', index=False)
            print(f"   • Unmatched authors sheet: {len(unmatched_df)} entries")
    
    print(f"\n✅ Excel file created: {excel_filename}")
    print("\nFile contains:")
    for conf_name, results_df in conference_results.items():
        if not results_df.empty:
            print(f"   • {conf_name}_2026 sheet: {len(results_df)} matched participants with full profiles")
    print(f"   • Summary sheet: Conference comparison and statistics")
    if all_unmatched_results:
        print(f"   • Unmatched_Authors sheet: {len(all_unmatched_results)} entries for manual review")
    
    # Print overall statistics
    if all_matched_results:
        total_predicted = sum(len(conferences_data[conf]) for conf in conferences_data if not conferences_data[conf].empty)
        total_matched = len(all_matched_results)
        overall_match_rate = (total_matched / total_predicted * 100) if total_predicted > 0 else 0
        
        print(f"\n🎯 OVERALL STATISTICS:")
        print(f"   • Total predicted participants across all conferences: {total_predicted}")
        print(f"   • Total successfully matched with scholar profiles: {total_matched}")
        print(f"   • Overall match rate: {overall_match_rate:.1f}%")
        print(f"   • Total unmatched for manual review: {len(all_unmatched_results)}")
    
    print(f"\n🎉 Scholar profile matching complete!")
    print("Use the Excel file to review predicted participants with their complete academic profiles.")

if __name__ == "__main__":
    main()
