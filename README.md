# Scholar Scraper

Scholar Scraper is a Python-based tool designed to collect, process, and visualize academic profile data from sources such as ICLR and NeurIPS. The project includes scripts for scraping, data aggregation, and a Streamlit dashboard for interactive data exploration.

## Features
- Scrapes and aggregates academic profiles from major conferences (ICLR, NeurIPS)
- Stores data in Parquet and CSV formats for efficient analysis
- Provides a Streamlit dashboard for data visualization and exploration
- Progress tracking and queue management for scraping tasks

## Project Structure
- `main.py`: Main script for running the scraper or data processing tasks
- `dashboard.py`: Streamlit dashboard for visualizing and exploring the collected data
- `iclr_2020_2025_combined_data.parquet`, `neurips_2020_2024_combined_data.parquet`: Aggregated data files
- `queue.txt`: Queue of profiles or tasks to be processed
- `scholar_profiles_progressssss.csv`: Progress tracking for scraping
- `requirements.txt`: Python dependencies
- `.devcontainer/`, `.streamlit/`: Configuration files for development and dashboard

## Getting Started
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the scraper:**
   ```bash
   python main.py
   ```
3. **Launch the dashboard:**
   ```bash
   streamlit run dashboard.py
   ```

## Requirements
- Python 3.8+
- See `requirements.txt` for full list

## License
MIT License

## Author
krishnan
