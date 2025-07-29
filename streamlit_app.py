import streamlit as st
import os
from selenium import webdriver
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.common.exceptions import WebDriverException, SessionNotCreatedException
from webdriver_manager.firefox import GeckoDriverManager
import time

st.set_page_config(layout="wide", page_title="Selenium Test", page_icon="🧪")

st.title("Selenium WebDriver Test on Streamlit Cloud")

# --- Set environment variable for tokenizers (important for Hugging Face models) ---
# This is from your original code, include it just in case it's a silent dependency
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@st.cache_resource(show_spinner="Initializing Selenium WebDriver...")
def get_selenium_driver():
    st.info("Attempting to get Selenium driver...")
    firefox_options = FirefoxOptions()
    firefox_options.add_argument("--headless")
    firefox_options.add_argument("--no-sandbox")
    firefox_options.add_argument("--disable-dev-shm-usage")
    firefox_options.add_argument("--window-size=1920,1080") # Add a window size for headless

    try:
        st.info("Attempting to install geckodriver via WebDriverManager...")
        # This line will download and set up geckodriver if not present
        service = FirefoxService(GeckoDriverManager().install())
        st.success(f"Geckodriver installed/found at: {service.path}")

        st.info("Attempting to initialize Firefox WebDriver with options...")
        driver = webdriver.Firefox(options=firefox_options, service=service)
        st.success("Firefox WebDriver initialized successfully!")
        return driver
    except SessionNotCreatedException as e:
        st.error(f"Failed to create Selenium session: {e}")
        st.stop() # Stop Streamlit execution
    except WebDriverException as e:
        st.error(f"Failed to initialize Firefox WebDriver (WebDriverException): {e}")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during WebDriver setup: {e}")
        st.stop()

# --- UI to trigger the driver initialization ---
st.write("Click the button below to attempt to initialize the Selenium Firefox WebDriver.")

if st.button("Initialize Selenium Driver"):
    driver_status_placeholder = st.empty()
    driver_status_placeholder.info("Button clicked. Attempting to initialize driver...")

    try:
        driver = get_selenium_driver()

        # If driver initialization was successful, try a simple page load
        if driver:
            driver_status_placeholder.success("WebDriver initialized. Attempting to get example.com...")
            try:
                driver.get("http://example.com")
                time.sleep(2) # Give it a moment to load
                page_title = driver.title
                driver_status_placeholder.success(f"Successfully loaded example.com! Page title: {page_title}")
            except Exception as e:
                driver_status_placeholder.error(f"Error navigating to example.com: {e}")
            finally:
                # IMPORTANT: Close the driver when done to free up resources
                st.info("Closing WebDriver...")
                driver.quit()
                st.info("WebDriver closed.")
        else:
            driver_status_placeholder.error("WebDriver initialization failed (driver object is None).")

    except Exception as e:
        # This outer try-except catches anything not handled by the inner ones in get_selenium_driver
        driver_status_placeholder.error(f"An unhandled error occurred after button click: {e}")
        if 'driver' in locals() and driver:
            driver.quit()

st.markdown("---")
st.info("Check Streamlit Cloud logs for detailed output.")
