import streamlit as st
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import WebDriverException, SessionNotCreatedException
from webdriver_manager.chrome import ChromeDriverManager
import time

st.set_page_config(layout="wide", page_title="Selenium Chrome Test", page_icon="🧪")

st.title("Selenium Chrome WebDriver Test on Streamlit Cloud")

# --- Set environment variable for tokenizers (important for Hugging Face models) ---
# This is from your original code, keeping it for consistency, though not directly related to Selenium
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Use st.cache_resource for caching heavy objects like WebDriver instances
# This ensures the driver is initialized only once per Streamlit session
@st.cache_resource(show_spinner="Initializing Selenium WebDriver...")
def get_chrome_driver():
    st.info("Attempting to get Chrome driver...")
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode (no GUI)
    chrome_options.add_argument("--no-sandbox")  # Required for running in containerized environments
    chrome_options.add_argument("--disable-dev-shm-usage")  # Overcomes limited resource problems
    chrome_options.add_argument("--window-size=1920,1080") # Set a default window size for headless
    chrome_options.add_argument("--disable-gpu") # Disable GPU hardware acceleration

    try:
        st.info("Attempting to install chromedriver via WebDriverManager...")
        # This line will download and set up chromedriver if not present
        service = Service(ChromeDriverManager().install())
        st.success(f"Chromedriver installed/found at: {service.path}")

        st.info("Attempting to initialize Chrome WebDriver with options...")
        driver = webdriver.Chrome(service=service, options=chrome_options)
        st.success("Chrome WebDriver initialized successfully!")
        return driver
    except SessionNotCreatedException as e:
        st.error(f"Failed to create Selenium session: {e}")
        st.stop() # Stop Streamlit execution if session cannot be created
    except WebDriverException as e:
        st.error(f"Failed to initialize Chrome WebDriver (WebDriverException): {e}")
        st.stop() # Stop Streamlit execution on WebDriver errors
    except Exception as e:
        st.error(f"An unexpected error occurred during WebDriver setup: {e}")
        st.stop() # Catch any other unexpected errors

# --- UI to trigger the driver initialization and a simple test ---
st.write("Click the button below to attempt to initialize the Selenium Chrome WebDriver and load example.com.")

if st.button("Initialize Selenium Driver and Test"):
    driver_status_placeholder = st.empty()
    driver_status_placeholder.info("Button clicked. Attempting to initialize driver...")
    
    driver = None # Initialize driver to None
    try:
        driver = get_chrome_driver()
        
        # If driver initialization was successful, try a simple page load
        if driver:
            driver_status_placeholder.success("WebDriver initialized. Attempting to get http://example.com...")
            try:
                driver.get("http://example.com")
                time.sleep(2) # Give it a moment to load
                page_title = driver.title
                page_source = driver.page_source
                driver_status_placeholder.success(f"Successfully loaded example.com! Page title: {page_title}")
                st.subheader("Page Source of example.com:")
                st.code(page_source)
            except Exception as e:
                driver_status_placeholder.error(f"Error navigating to example.com: {e}")
            finally:
                # IMPORTANT: Close the driver when done to free up resources
                if driver:
                    st.info("Closing WebDriver...")
                    driver.quit()
                    st.info("WebDriver closed.")
        else:
            driver_status_placeholder.error("WebDriver initialization failed (driver object is None).")

    except Exception as e:
        # This outer try-except catches anything not handled by the inner ones in get_chrome_driver
        driver_status_placeholder.error(f"An unhandled error occurred after button click: {e}")
        if driver: # Ensure driver is closed even if outer error occurs
            driver.quit()

st.markdown("---")
st.info("Check Streamlit Cloud logs for detailed output. If this test works, the environment setup is correct.")
