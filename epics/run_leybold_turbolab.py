#!/usr/bin/env python3

import sys
import time
import argparse
import socket
import json
import logging
import subprocess
import os
from datetime import datetime

# Set EPICS environment variables for proper connection
# Don't specify port - let EPICS auto-discover using broadcast
os.environ['EPICS_CA_ADDR_LIST'] = '127.0.0.1'  # Use IP instead of hostname
os.environ['EPICS_CA_AUTO_ADDR_LIST'] = 'NO'
os.environ['EPICS_CA_SERVER_PORT'] = '5064'  
os.environ['EPICS_CA_MAX_ARRAY_BYTES'] = '16384'
os.environ['EPICS_CA_CONN_TMO'] = '10.0'

import epics
from epics import PV, caput
import requests
from bs4 import BeautifulSoup
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('leybold_turbolab.log')
    ]
)
logger = logging.getLogger('leybold_turbolab')

class LeyboldTurbolab:
    """
    Interface to Leybold Turbolab pumping station.
    Scrapes the web interface at the pumping station's IP address.
    """
    
    # Default port for the web interface
    DEFAULT_PORT = 80
    
    # Register map for Leybold Turbolab (keeping this to maintain compatibility)
    REGISTERS = {
        # Pump status registers
        "turbo_pump_speed": {"addr": 0x1000, "type": "float", "scale": 1.0, "unit": "rpm"},
        "turbo_pump_current": {"addr": 0x1002, "type": "float", "scale": 0.1, "unit": "A"},
        "turbo_pump_power": {"addr": 0x1004, "type": "float", "scale": 1.0, "unit": "W"},
        "turbo_pump_drive_temp": {"addr": 0x1006, "type": "float", "scale": 0.1, "unit": "°C"},
        "turbo_pump_bearing_temp": {"addr": 0x1008, "type": "float", "scale": 0.1, "unit": "°C"},
        "turbo_pump_status": {"addr": 0x1010, "type": "uint16", "scale": 1.0, "unit": ""},
        
        # Backing pump registers
        "backing_pump_speed": {"addr": 0x1100, "type": "float", "scale": 1.0, "unit": "rpm"},
        "backing_pump_current": {"addr": 0x1102, "type": "float", "scale": 0.1, "unit": "A"},
        "backing_pump_power": {"addr": 0x1104, "type": "float", "scale": 1.0, "unit": "W"},
        "backing_pump_temp": {"addr": 0x1106, "type": "float", "scale": 0.1, "unit": "°C"},
        "backing_pump_status": {"addr": 0x1110, "type": "uint16", "scale": 1.0, "unit": ""},
        
        # Pressure gauges
        "inlet_pressure": {"addr": 0x1200, "type": "float", "scale": 1.0, "unit": "mbar"},
        "foreline_pressure": {"addr": 0x1202, "type": "float", "scale": 1.0, "unit": "mbar"},
        "chamber_pressure": {"addr": 0x1204, "type": "float", "scale": 1.0, "unit": "mbar"},
        
        # System status
        "system_status": {"addr": 0x1300, "type": "uint16", "scale": 1.0, "unit": ""},
        "error_code": {"addr": 0x1302, "type": "uint16", "scale": 1.0, "unit": ""},
        "warning_code": {"addr": 0x1304, "type": "uint16", "scale": 1.0, "unit": ""},
        
        # Operating hours
        "turbo_pump_hours": {"addr": 0x1400, "type": "uint32", "scale": 1.0, "unit": "h"},
        "backing_pump_hours": {"addr": 0x1402, "type": "uint32", "scale": 1.0, "unit": "h"},
    }
    
    # Status codes for interpretation
    STATUS_CODES = {
        0: "Off",
        1: "Starting",
        2: "Normal operation",
        3: "Stopping",
        4: "Fault",
        5: "Maintenance required"
    }
    
    def __init__(self, host, port=DEFAULT_PORT, timeout=5.0):
        """Initialize connection to the Leybold Turbolab pumping station."""
        self.host = host
        self.port = port
        self.timeout = timeout
        self.session = None
        self.last_data = {}
        self.simulation_mode = False
        
        # Try to connect, fall back to simulation if connection fails
        if not self.connect():
            logger.warning("Falling back to simulation mode")
            self.simulation_mode = True
    
    def connect(self):
        """Establish connection to the pumping station web interface."""
        try:
            self.session = requests.Session()
            self.session.timeout = self.timeout
            # Add headers to mimic a browser
            self.session.headers.update({
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
            })
            
            # Test the connection with a request to the root URL
            url = f"http://{self.host}"
            logger.info(f"Attempting to connect to {url}")
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()  # Raise an exception for HTTP errors
            
            # Log response details for debugging
            logger.info(f"Connected to {url} - Status: {response.status_code}")
            logger.info(f"Response headers: {response.headers}")
            logger.info(f"Response content type: {response.headers.get('Content-Type', 'unknown')}")
            logger.info(f"Response length: {len(response.text)} bytes")
            
            return True
        except Exception as e:
            logger.error(f"Failed to connect to web interface at {self.host}: {str(e)}")
            # Try to provide more detailed error information
            if hasattr(e, 'response') and e.response:
                logger.error(f"Error status code: {e.response.status_code}")
                logger.error(f"Error content: {e.response.text[:500]}...")
            self.session = None
            return False
    
    def disconnect(self):
        """Close the connection."""
        if self.session:
            self.session.close()
            self.session = None
            logger.info("Disconnected from Leybold Turbolab web interface")
    
    def _login(self):
        """
        Log in to the web interface by submitting credentials directly to the login endpoint.

        Returns:
            bool: True if login is successful, False otherwise.
        """
        try:
            # Construct the login URL with credentials
            login_url = f"http://{self.host}/?login=useruser"
            logger.info(f"Attempting to log in at {login_url}")

            # Send the login request
            response = self.session.get(login_url, timeout=self.timeout)
            response.raise_for_status()

            # Log the response content for debugging
            logger.debug(f"Login response content: {response.text[:500]}...")

            # Check if login was successful by inspecting the response
            if "logout" in response.text.lower():
                logger.info("Login successful.")
                return True
            else:
                logger.error("Login failed. Check credentials or login endpoint.")
                return False

        except Exception as e:
            logger.error(f"Error during login: {str(e)}")
            return False

    def scrape_web_data(self):
        """
        Scrape data from the pumping station's web interface.
        
        Returns:
            dict: Dictionary with parameter names as keys and their values
        """
        if self.simulation_mode:
            return self._generate_simulated_values()
            
        if not self.session:
            if not self.connect():
                logger.error("Failed to connect to web interface for scraping")
                return {}

        # Try Selenium scraping first as it's the most reliable for JavaScript-heavy pages
        data = {}
        selenium_success = False
        
        logger.info("Attempting to scrape using Selenium WebDriver first")
        if self._try_selenium_scrape(data):
            selenium_success = True
            logger.info("Successfully extracted data using Selenium WebDriver")
            
            # If we got enough data points from Selenium, return the data
            if len(data) >= 3:
                # Determine system status based on available data
                if "turbo_pump_speed" in data and data["turbo_pump_speed"] > 1000:
                    data["turbo_pump_status"] = 2  # Normal operation
                    data["turbo_pump_status_text"] = self.STATUS_CODES[2]
                else:
                    data["turbo_pump_status"] = 0  # Off
                    data["turbo_pump_status_text"] = self.STATUS_CODES[0]
                    
                # Determine overall system status based on turbo pump status
                data["system_status"] = data.get("turbo_pump_status", 0)
                
                # Fill in any missing values with defaults
                self._fill_missing_values(data)
                
                logger.info(f"Successfully scraped data using Selenium: {len(data)} values")
                self.last_data = data
                return data
        else:
            logger.info("Selenium scraping was not successful, falling back to other methods")

        # First try direct access to the homeval container without login
        direct_access_success = False
        
        # Try accessing /0.hgz page directly first, which often contains the homeval container
        #if self._scrape_hgz_page(data):
        #    direct_access_success = True
        #    logger.info("Successfully extracted data from /0.hgz page without login")
        
        # If we couldn't get data from the /0.hgz page, try the main page
        #if not direct_access_success and self._scrape_main_page(data):
        #    direct_access_success = True
        #    logger.info("Successfully extracted data from main page without login")
            
        # If direct access worked and we got enough data points, return the data
        if direct_access_success and len(data) >= 3:
            # Determine system status based on available data
            if "turbo_pump_speed" in data and data["turbo_pump_speed"] > 1000:
                data["turbo_pump_status"] = 2  # Normal operation
                data["turbo_pump_status_text"] = self.STATUS_CODES[2]
            else:
                data["turbo_pump_status"] = 0  # Off
                data["turbo_pump_status_text"] = self.STATUS_CODES[0]
                
            # Determine overall system status based on turbo pump status
            data["system_status"] = data.get("turbo_pump_status", 0)
            
            # Fill in any missing values with defaults
            self._fill_missing_values(data)
            
            logger.info(f"Successfully scraped data from web interface without login: {len(data)} values")
            self.last_data = data
            return data
        
        # If direct access failed, try logging in
        #logger.info("Direct access without login didn't provide enough data, attempting login...")
        #if not self._login():
        #    logger.warning("Failed to log in, but will continue with scraping attempts")

        # Attempt to scrape the data viewing page first
        #data = self._scrape_data_viewing_page()
        success = len(data) > 0

        # If data viewing page scraping fails, fall back to other methods
        if not success:
            # Try scraping the trend table
            #data = self._scrape_trend_table()
            success = len(data) > 0

            # If trend table scraping fails, try other methods
            if not success:
                # Try direct main page
                #if self._scrape_main_page(data):
                #    success = True

                # If first approach didn't yield enough data, try other paths
                if not success or len(data) < 3:
                    paths = ['/', '/status', '/data', '/overview', '/index.htm', '/index.html']
                    for path in paths:
                        if path == '/' and not success:  # Skip if we already tried the main page
                            continue
                        if self._scrape_page(path, data):
                            success = True
                            break

                # If we still don't have data, try AJAX endpoints if they exist
                if not success or len(data) < 3:
                    ajax_paths = ['/ajax/status', '/api/data', '/get_values.cgi']
                    for path in ajax_paths:
                        if self._scrape_ajax(path, data):
                            success = True
                            break

                # If all direct approaches failed, try a broader extraction
                if not success or len(data) < 3:
                    if self._extract_all_numeric_data(data):
                        success = True

                # If we still don't have data, try the alternative scrape method
                if not success or len(data) < 3:
                    alternative_data = self._alternative_scrape()
                    if alternative_data:
                        data.update(alternative_data)
                        success = True

                # Explore additional endpoints for trend table
                if not success or len(data) < 3:
                    additional_data = self._scrape_additional_endpoints()
                    if additional_data:
                        data.update(additional_data)
                        success = True

        # If we got any data, process and return it
        if data:
            # Determine system status based on available data
            if "turbo_pump_speed" in data and data["turbo_pump_speed"] > 1000:
                data["turbo_pump_status"] = 2  # Normal operation
                data["turbo_pump_status_text"] = self.STATUS_CODES[2]
            else:
                data["turbo_pump_status"] = 0  # Off
                data["turbo_pump_status_text"] = self.STATUS_CODES[0]
                
            # Determine overall system status based on turbo pump status
            data["system_status"] = data.get("turbo_pump_status", 0)
            
            # Fill in any missing values with defaults
            self._fill_missing_values(data)
            
            logger.info(f"Successfully scraped data from web interface: {len(data)} values")
            self.last_data = data
            return data
        else:
            logger.error("Failed to extract any data using all available methods")
            return self._generate_simulated_values()

    def _try_selenium_scrape(self, data):
        """
        Try to scrape using Selenium WebDriver which allows proper JavaScript execution.
        This method is crucial for extracting values from JavaScript-driven interfaces.
        
        Args:
            data: Dictionary to store extracted data
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if selenium is installed
            try:
                from selenium import webdriver
                from selenium.webdriver.chrome.options import Options
                from selenium.webdriver.common.by import By
                from selenium.webdriver.support.ui import WebDriverWait
                from selenium.webdriver.support import expected_conditions as EC
                from selenium.webdriver.chrome.service import Service
                from selenium.common.exceptions import TimeoutException, WebDriverException
            except ImportError:
                logger.warning("Selenium not installed, cannot use browser automation")
                logger.info("Install with: pip install selenium")
                return False
                
            logger.info("Starting Selenium WebDriver for browser automation")
            
            # Configure Chrome options
            chrome_options = Options()
            chrome_options.add_argument("--headless")  # Run in background
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")  # Set larger window size
            chrome_options.add_argument("--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36")
            
            try:
                # Try to start the browser
                browser = webdriver.Chrome(options=chrome_options)
                
                # Set timeout
                browser.set_page_load_timeout(30)
                
                # Load the page
                url = f"http://{self.host}"
                logger.info(f"Loading {url} with Selenium")
                
                try:
                    browser.get(url)
                except TimeoutException:
                    logger.warning("Page load timed out, but continuing anyway as the DOM might be usable")
                
                # Wait for page to load and JavaScript to execute
                logger.info("Waiting for page to load and JavaScript to execute")
                try:
                    WebDriverWait(browser, 10).until(
                        EC.presence_of_element_located((By.TAG_NAME, "body"))
                    )
                except TimeoutException:
                    logger.warning("Timed out waiting for body element, but continuing")
                
                # Additional wait to allow JavaScript to update values - IMPORTANT for non-zero values
                import time
                logger.info("Waiting 1 seconds for JavaScript to fully initialize values...")
                time.sleep(1)
                
                # Try to login if login form is present
                try:
                    login_inputs = browser.find_elements(By.XPATH, "//input[@type='password' or @type='text' and contains(@id, 'user')]")
                    login_buttons = browser.find_elements(By.XPATH, "//button[contains(text(), 'Login') or contains(@id, 'login')] | //input[@type='submit' and contains(@value, 'Login')]")
                    
                    if login_inputs and login_buttons:
                        logger.info("Found login form, attempting to login")
                        
                        # Try with default credentials
                        for input_field in login_inputs:
                            if 'user' in input_field.get_attribute('id').lower():
                                input_field.send_keys('user')
                            elif 'password' in input_field.get_attribute('id').lower():
                                input_field.send_keys('user')
                        
                        # Click the login button
                        login_buttons[0].click()
                        
                        # Wait after login
                        logger.info("Waiting after login...")
                        time.sleep(5)
                except Exception as login_err:
                    logger.warning(f"Login attempt failed: {login_err}")
                
                # Get the page source after JavaScript has executed
                page_source = browser.page_source
                
                # Save for debugging
                with open("/tmp/leybold_selenium_page.html", "w") as f:
                    f.write(page_source)
                
                # Dump all element IDs and text to help with debugging
                try:
                    logger.info("Dumping key elements for debugging:")
                    all_elements = browser.find_elements(By.XPATH, "//*[@id]")
                    for element in all_elements[:20]:  # Limit to first 20 to avoid log flooding
                        element_id = element.get_attribute('id')
                        element_text = element.text.strip() if element.text else "[No text]"
                        logger.info(f"Element ID: {element_id}, Text: {element_text}")
                except Exception as dump_err:
                    logger.warning(f"Error dumping elements: {dump_err}")
                
                # Improved extraction with better selectors and error handling
                success = False
                
                # Extract bearing temperature - expanded selectors
                bearing_temp_elements = browser.find_elements(By.XPATH, 
                    "//*[contains(text(), '°C') or contains(@id, 'bear') or contains(@id, 'temp') or contains(@class, 'bearing') or contains(@id, '20v')]")
                
                for element in bearing_temp_elements:
                    try:
                        # Try to get text first
                        text = element.text.strip()
                        # If no text, try to get value attribute
                        if not text:
                            text = element.get_attribute('value')
                            if not text:
                                text = element.get_attribute('textContent')
                        
                        if text and ('°C' in text or text.replace('.', '', 1).isdigit()):
                            logger.info(f"Found bearing temperature element with text: {text}")
                            # Extract numeric value
                            import re
                            # Try to find a number followed by °C
                            matches = re.search(r'(\d+\.?\d*)\s*°C', text)
                            # If that fails, try to find any number
                            if not matches:
                                matches = re.search(r'(\d+\.?\d*)', text)
                            
                            if matches:
                                value = float(matches.group(1))
                                logger.info(f"Extracted bearing temperature: {value} °C")
                                data["turbo_pump_bearing_temp"] = value
                                success = True
                                break
                    except Exception as e:
                        logger.error(f"Error processing bearing temperature element: {e}")
                
                # Try checking IDs directly for known patterns - this is often more reliable
                known_ids = [
                    "20v",  # Bearing temperature
                    "72v",  # Chamber pressure
                    "73v",  # Foreline pressure
                    "pwr",  # Power
                    "rpm",  # Speed
                    "curr"  # Current
                ]
                
                # Specifically look for the exponent elements
                exponent_ids = {
                    "72p": "72v",  # Chamber pressure exponent
                    "73p": "73v",  # Foreline pressure exponent
                }
                
                for element_id in known_ids:
                    try:
                        elements = browser.find_elements(By.ID, element_id)
                        if elements:
                            text = elements[0].text.strip()
                            if not text:
                                text = elements[0].get_attribute('value')
                            if not text:
                                text = elements[0].get_attribute('textContent')
                                
                            logger.info(f"Found element with ID {element_id}: {text}")
                            
                            # Process based on ID type
                            if element_id == "20v" and text:
                                # Bearing temperature
                                import re
                                matches = re.search(r'(\d+\.?\d*)', text)
                                if matches:
                                    value = float(matches.group(1))
                                    logger.info(f"Extracted bearing temperature from ID {element_id}: {value} °C")
                                    data["turbo_pump_bearing_temp"] = value
                                    success = True
                            
                            elif element_id in ["72v", "73v"] and text:
                                # Chamber or Foreline pressure - handle mantissa part
                                # First attempt to extract just the mantissa
                                import re
                                mantissa_match = re.search(r'(\d+\.?\d*)', text)
                                if mantissa_match:
                                    mantissa = float(mantissa_match.group(1))
                                    logger.info(f"Extracted mantissa from ID {element_id}: {mantissa}")
                                    
                                    # Now look for the corresponding exponent
                                    exponent_id = None
                                    if element_id == "72v":
                                        exponent_id = "72p"  # Chamber pressure exponent
                                    elif element_id == "73v":
                                        exponent_id = "73p"  # Foreline pressure exponent
                                    
                                    if exponent_id:
                                        exponent_elements = browser.find_elements(By.ID, exponent_id)
                                        if exponent_elements:
                                            exp_text = exponent_elements[0].text.strip()
                                            if not exp_text:
                                                exp_text = exponent_elements[0].get_attribute('value')
                                            if not exp_text:
                                                exp_text = exponent_elements[0].get_attribute('textContent')
                                            
                                            logger.info(f"Found exponent element with ID {exponent_id}: {exp_text}")
                                            
                                            # Try to extract exponent value
                                            exp_match = re.search(r'[-+]?\d+', exp_text)
                                            if exp_match:
                                                exponent = int(exp_match.group(0))
                                            else:
                                                # Handle superscript characters
                                                exponent = self._convert_superscript_to_int(exp_text)
                                            
                                            # Calculate the final value
                                            value = mantissa * (10 ** exponent)
                                            
                                            logger.info(f"Calculated value for {element_id} with exponent {exponent}: {value}")
                                            
                                            if element_id == "72v":
                                                data["chamber_pressure"] = value
                                                logger.info(f"Extracted chamber pressure: {value} mbar")
                                            elif element_id == "73v":
                                                data["foreline_pressure"] = value
                                                logger.info(f"Extracted foreline pressure: {value} mbar")
                                            
                                            success = True
                                        else:
                                            # If exponent not found, try legacy method
                                            if "×" in text or "x" in text or "10" in text:
                                                # Try to parse scientific notation all in one element
                                                sci_match = re.search(r'(\d+\.?\d*)\s*[×x]\s*10([¹²³⁴⁵⁶⁷⁸⁹⁰-]|\d+)', text)
                                                if sci_match:
                                                    base = float(sci_match.group(1))
                                                    exp_part = sci_match.group(2)
                                                    exp = self._convert_superscript_to_int(exp_part)
                                                    value = base * (10 ** exp)
                                                    
                                                    if element_id == "72v":
                                                        data["chamber_pressure"] = value
                                                        logger.info(f"Extracted chamber pressure using legacy method: {value} mbar")
                                                    elif element_id == "73v":
                                                        data["foreline_pressure"] = value
                                                        logger.info(f"Extracted foreline pressure using legacy method: {value} mbar")
                                                    
                                                    success = True
                                            else:
                                                # If no indication of scientific notation, use the value as is
                                                if element_id == "72v":
                                                    data["chamber_pressure"] = mantissa
                                                    logger.info(f"Using mantissa as chamber pressure: {mantissa} mbar")
                                                elif element_id == "73v":
                                                    data["foreline_pressure"] = mantissa
                                                    logger.info(f"Using mantissa as foreline pressure: {mantissa} mbar")
                                                success = True
                    except Exception as id_err:
                        logger.error(f"Error processing element with ID {element_id}: {id_err}")
                
                # Try directly accessing the exponent elements
                for exp_id, value_id in exponent_ids.items():
                    try:
                        # Check if we already have the corresponding value
                        if (value_id == "72v" and "chamber_pressure" in data) or (value_id == "73v" and "foreline_pressure" in data):
                            continue
                        
                        # Otherwise, try to look up both parts directly
                        value_elements = browser.find_elements(By.ID, value_id)
                        exp_elements = browser.find_elements(By.ID, exp_id)
                        
                        if value_elements and exp_elements:
                            value_text = value_elements[0].text.strip() or value_elements[0].get_attribute('textContent')
                            exp_text = exp_elements[0].text.strip() or exp_elements[0].get_attribute('textContent')
                            
                            logger.info(f"Found mantissa with ID {value_id}: {value_text}")
                            logger.info(f"Found exponent with ID {exp_id}: {exp_text}")
                            
                            if value_text and exp_text:
                                import re
                                mantissa_match = re.search(r'(\d+\.?\d*)', value_text)
                                if mantissa_match:
                                    mantissa = float(mantissa_match.group(1))
                                    
                                    # Parse exponent
                                    exp_match = re.search(r'[-+]?\d+', exp_text)
                                    if exp_match:
                                        exponent = int(exp_match.group(0))
                                    else:
                                        exponent = self._convert_superscript_to_int(exp_text)
                                    
                                    # Calculate the full value
                                    value = mantissa * (10 ** exponent)
                                    
                                    if value_id == "72v":
                                        data["chamber_pressure"] = value
                                        logger.info(f"Extracted chamber pressure from direct ID lookup: {value} mbar")
                                    elif value_id == "73v":
                                        data["foreline_pressure"] = value
                                        logger.info(f"Extracted foreline pressure from direct ID lookup: {value} mbar")
                                    
                                    success = True
                    except Exception as exp_err:
                        logger.error(f"Error processing exponent with ID {exp_id}: {exp_err}")
                
                # Try accessing the /0.hgz page which often has more direct values
                try:
                    logger.info("Trying to access /0.hgz page via Selenium")
                    browser.get(f"http://{self.host}/0.hgz")
                    try:
                        WebDriverWait(browser, 10).until(
                            EC.presence_of_element_located((By.TAG_NAME, "body"))
                        )
                    except TimeoutException:
                        logger.warning("Timed out waiting for body element, but continuing")

                    time.sleep(1)  # Wait for page to load
                    
                    # Save page source for debugging
                    with open("/tmp/leybold_selenium_hgz_page.html", "w") as f:
                        f.write(browser.page_source)
                    
                    # Try to find elements again on this page, using same logic as above
                    for element_id in ["20v", "72v", "73v"]:
                        try:
                            elements = browser.find_elements(By.ID, element_id)
                            if elements:
                                text = elements[0].text.strip() or elements[0].get_attribute('textContent')
                                logger.info(f"Found element with ID {element_id} on /0.hgz page: {text}")
                                
                                # Process the same way as above
                                if element_id == "20v" and text:
                                    import re
                                    matches = re.search(r'(\d+\.?\d*)', text)
                                    if matches:
                                        value = float(matches.group(1))
                                        data["turbo_pump_bearing_temp"] = value
                                        success = True
                                elif element_id in ["72v", "73v"] and text:
                                    # Check for corresponding exponent element
                                    exponent_id = f"{element_id[0:2]}p"
                                    exponent_elements = browser.find_elements(By.ID, exponent_id)
                                    
                                    import re
                                    mantissa_match = re.search(r'(\d+\.?\d*)', text)
                                    if mantissa_match:
                                        mantissa = float(mantissa_match.group(1))
                                        
                                        if exponent_elements:
                                            exp_text = exponent_elements[0].text.strip() or exponent_elements[0].get_attribute('textContent')
                                            logger.info(f"Found exponent element with ID {exponent_id} on /0.hgz page: {exp_text}")
                                            
                                            # Try to extract exponent value
                                            exp_match = re.search(r'[-+]?\d+', exp_text)
                                            if exp_match:
                                                exponent = int(exp_match.group(0))
                                            else:
                                                exponent = self._convert_superscript_to_int(exp_text)
                                            
                                            # Calculate the final value
                                            value = mantissa * (10 ** exponent)
                                        else:
                                            # Fall back to simple number extraction
                                            value = mantissa
                                        
                                        if element_id == "72v":
                                            data["chamber_pressure"] = value
                                        elif element_id == "73v":
                                            data["foreline_pressure"] = value
                                        
                                        success = True
                        except Exception as hgz_err:
                            logger.error(f"Error processing element with ID {element_id} on /0.hgz page: {hgz_err}")
                except Exception as hgz_page_err:
                    logger.error(f"Error accessing /0.hgz page via Selenium: {hgz_page_err}")
                
                # Try to take a screenshot for debugging
                try:
                    logger.info("Taking screenshot of the page")
                    browser.save_screenshot("/tmp/leybold_screenshot.png")
                    logger.info("Screenshot saved to /tmp/leybold_screenshot.png")
                except Exception as ss_err:
                    logger.error(f"Error taking screenshot: {ss_err}")
                
                # Try to extract data from JavaScript variables
                try:
                    logger.info("Trying to extract data from JavaScript variables")
                    js_variables = [
                        "window.gaugeData", "window.pumpData", "window.sensorData",
                        "window.pressureData", "window.statusData", "window.deviceData",
                        "window.globalData", "window.turbolabData", "window.systemData"
                    ]
                    
                    for js_var in js_variables:
                        try:
                            result = browser.execute_script(f"return {js_var};")
                            if result:
                                logger.info(f"Found data in {js_var}: {result}")
                                if isinstance(result, dict):
                                    # Extract relevant values based on key names
                                    for key, value in result.items():
                                        key_lower = key.lower()
                                        if isinstance(value, (int, float)) and value != 0:
                                            if "chamber" in key_lower or "gauge1" in key_lower:
                                                data["chamber_pressure"] = float(value)
                                                logger.info(f"Extracted chamber pressure from JS: {value} mbar")
                                                success = True
                                            elif "foreline" in key_lower or "gauge2" in key_lower or "backing" in key_lower:
                                                data["foreline_pressure"] = float(value)
                                                logger.info(f"Extracted foreline pressure from JS: {value} mbar")
                                                success = True
                                            elif "bearing" in key_lower and "temp" in key_lower:
                                                data["turbo_pump_bearing_temp"] = float(value)
                                                logger.info(f"Extracted bearing temperature from JS: {value} °C")
                                                success = True
                        except Exception as js_err:
                            logger.warning(f"Error accessing {js_var}: {js_err}")
                except Exception as js_ex:
                    logger.error(f"Error in JavaScript data extraction: {js_ex}")
                
                # Clean up
                browser.quit()
                
                # Return True if we found data
                return success
                
            except Exception as e:
                logger.error(f"Error using Selenium: {str(e)}")
                try:
                    browser.quit()
                except:
                    pass
                return False
                
        except Exception as e:
            logger.error(f"Error initializing Selenium: {str(e)}")
            return False

    def _convert_superscript_to_int(self, superscript_str):
        """
        Convert superscript characters to regular integers.
        Handles both Unicode superscripts and regular digits.
        
        Args:
            superscript_str: String containing superscript characters or regular digits
            
        Returns:
            int: Integer value of the superscript
        """
        # Map of superscript characters to their integer values
        superscript_map = {
            '⁰': 0, '¹': 1, '²': 2, '³': 3, '⁴': 4,
            '⁵': 5, '⁶': 6, '⁷': 7, '⁸': 8, '⁹': 9,
            '-': -1  # Handle negative exponent (⁻)
        }
        
        # If it's a regular digit string, convert directly
        if superscript_str.lstrip('-').isdigit():
            return int(superscript_str)
        
        # For superscript characters
        result = 0
        is_negative = False
        
        for char in superscript_str:
            if char == '⁻':
                is_negative = True
            elif char in superscript_map:
                result = result * 10 + superscript_map[char]
        
        return -result if is_negative else result

    def _fill_missing_values(self, data):
        """
        Fill in missing values with defaults or last known good values.
        
        Args:
            data: Dictionary of scraped data to be filled
        """
        # Define default values for keys that might be missing
        defaults = {
            "turbo_pump_speed": 0.0,
            "turbo_pump_current": 0.0,
            "turbo_pump_power": 0.0,
            "turbo_pump_drive_temp": 25.0,  # Room temperature as default
            "turbo_pump_bearing_temp": 25.0,  # Room temperature as default
            "turbo_pump_status": 0,  # Off
            "turbo_pump_status_text": self.STATUS_CODES[0],  # Off
            "backing_pump_speed": 0.0,
            "backing_pump_current": 0.0,
            "backing_pump_power": 0.0,
            "backing_pump_temp": 25.0,  # Room temperature as default
            "backing_pump_status": 0,  # Off
            "backing_pump_status_text": self.STATUS_CODES[0],  # Off
            "inlet_pressure": 1000.0,  # Atmospheric pressure as default
            "foreline_pressure": 1000.0,  # Atmospheric pressure as default
            "chamber_pressure": 1000.0,  # Atmospheric pressure as default
            "system_status": 0  # Off
        }
        
        # Use last known good values if available, otherwise use defaults
        for key, default_value in defaults.items():
            if key not in data:
                if key in self.last_data and self.last_data[key] is not None:
                    data[key] = self.last_data[key]
                else:
                    data[key] = default_value
        
        # Add status text based on status codes
        if "turbo_pump_status" in data and "turbo_pump_status_text" not in data:
            status_code = data["turbo_pump_status"]
            data["turbo_pump_status_text"] = self.STATUS_CODES.get(status_code, "Unknown")
            
        if "backing_pump_status" in data and "backing_pump_status_text" not in data:
            status_code = data["backing_pump_status"]
            data["backing_pump_status_text"] = self.STATUS_CODES.get(status_code, "Unknown")
            
        logger.debug(f"Filled missing values in data dictionary: {len(data)} total values")
        return data

def check_ioc_running():
    """Check if the EPICS IOC is running."""
    try:
        # Look for either auxioc or softIoc in the process list
        result = subprocess.run(['pgrep', '-f', 'softIoc|auxioc'], 
                                capture_output=True, text=True)
        return len(result.stdout.strip()) > 0
    except Exception as e:
        logger.warning(f"Error checking for IOC process: {e}")
        # If we can't check, assume it's running
        return True

def main():
    """Main function to run the Leybold Turbolab data acquisition loop."""
    parser = argparse.ArgumentParser(
        description='Interface with Leybold Turbolab pumping station and update EPICS records'
    )
    
    parser.add_argument(
        '--host',
        default='192.168.1.15',
        help='IP address of the Leybold Turbolab pumping station (default: 192.168.1.15)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=80,
        help='Web interface port (default: 80)'
    )
    parser.add_argument(
        '--interval',
        type=float,
        default=1.0,
        help='Update interval in seconds (default: 1.0)'
    )
    parser.add_argument(
        '--prefix',
        default='Y1:AUX-',
        help='Prefix for EPICS PVs (default: AUX:)'
    )
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Set logging level (default: INFO)'
    )
    parser.add_argument(
        '--simulate',
        action='store_true',
        help='Run in simulation mode without connecting to hardware'
    )
    
    args = parser.parse_args()
    
    # Set log level from command-line argument
    logger.setLevel(getattr(logging, args.log_level))
    
    # Check if the IOC is running
    if not check_ioc_running():
        logger.critical("EPICS IOC (auxioc) does not appear to be running. Please start it first.")
        logger.info("You can start the IOC with: ./start_auxioc.sh")
        sys.exit(1)
    
    # Create and connect to the Leybold pumping station
    leybold = LeyboldTurbolab(args.host, args.port)
    leybold.simulation_mode = args.simulate
    
    logger.info(f"Starting data acquisition loop with interval {args.interval}s")
    
    try:
        # Create mapping from registers to PV names
        pv_map = {}
        for reg_name, reg_info in LeyboldTurbolab.REGISTERS.items():
            pv_name = f"{args.prefix}{reg_name.upper()}"
            pv_map[reg_name] = pv_name
            logger.debug(f"Mapped register {reg_name} to PV {pv_name}")
            
        # Also map status text fields
        for reg_name in LeyboldTurbolab.REGISTERS:
            if "status" in reg_name:
                text_field = f"{reg_name}_text"
                pv_name = f"{args.prefix}{text_field.upper()}"
                pv_map[text_field] = pv_name
                logger.debug(f"Mapped register {text_field} to PV {pv_name}")
        
        while True:
            # Read all data from the pumping station using the scrape_web_data method
            start_time = time.time()
            data = leybold.scrape_web_data()
            
            if data:
                # Update EPICS PVs with the data
                update_epics_pvs(data, pv_map)
                
                # Optional: print summary of key values
                if 'turbo_pump_speed' in data:
                    logger.info(f"Turbo pump speed: {data['turbo_pump_speed']} rpm")
                if 'chamber_pressure' in data:
                    logger.info(f"Chamber pressure: {data['chamber_pressure']} mbar")
            else:
                logger.warning("Failed to read data from the pumping station")
                # Try to reconnect
                leybold.disconnect()
                leybold.connect()
            
            # Calculate wait time to maintain consistent interval
            elapsed = time.time() - start_time
            wait_time = max(0.1, args.interval - elapsed)
            time.sleep(wait_time)
            
    except KeyboardInterrupt:
        logger.info("Received interrupt, shutting down...")
    except Exception as e:
        logger.critical(f"Unexpected error: {str(e)}")
        import traceback
        logger.critical(traceback.format_exc())
    finally:
        leybold.disconnect()
        logger.info("Program terminated")


def update_epics_pvs(data, pv_map):
    """
    Update EPICS PVs with data from the pumping station.
    
    Args:
        data: Dictionary of parameter names and values
        pv_map: Dictionary mapping parameter names to PV names
    """
    # Create and cache PV objects to avoid constant reconnection
    if not hasattr(update_epics_pvs, 'pv_objects'):
        update_epics_pvs.pv_objects = {}
        update_epics_pvs.connection_attempts = {}
    
    updated_count = 0
    connection_retry_limit = 5  # Retry connecting to a PV this many times before giving up
    
    # If we haven't successfully updated any PVs in the last several calls,
    # try to recreate all PV objects
    if hasattr(update_epics_pvs, 'consecutive_failures'):
        if update_epics_pvs.consecutive_failures > 10:
            logger.warning("Too many consecutive failures, recreating all PV objects...")
            update_epics_pvs.pv_objects = {}
            update_epics_pvs.connection_attempts = {}
            update_epics_pvs.consecutive_failures = 0
    else:
        update_epics_pvs.consecutive_failures = 0
    
    for param_name, value in data.items():
        if param_name in pv_map:
            pv_name = pv_map[param_name]
            
            # Initialize connection attempt counter for this PV if it doesn't exist
            if pv_name not in update_epics_pvs.connection_attempts:
                update_epics_pvs.connection_attempts[pv_name] = 0
            
            # Skip PVs that have exceeded the retry limit
            if update_epics_pvs.connection_attempts[pv_name] >= connection_retry_limit:
                continue
            
            # Create or get the PV object
            try:
                if pv_name not in update_epics_pvs.pv_objects:
                    logger.info(f"Creating PV object for {pv_name}")
                    # Use auto_monitor=False for write-only PVs to reduce network traffic
                    pv_obj = PV(pv_name, connection_timeout=3.0, auto_monitor=False)
                    update_epics_pvs.pv_objects[pv_name] = pv_obj
                else:
                    pv_obj = update_epics_pvs.pv_objects[pv_name]
                
                # Check connection status and try to connect if needed
                if not pv_obj.connected:
                    logger.debug(f"Attempting to connect to {pv_name}")
                    pv_obj.connect(timeout=2.0)
                    update_epics_pvs.connection_attempts[pv_name] += 1
                
                # Try to update the PV if connected
                if pv_obj.connected:
                    pv_obj.put(value, timeout=1.0, wait=False)  # Non-blocking put
                    updated_count += 1
                    logger.debug(f"Updated PV {pv_name} = {value}")
                    # Reset the connection attempt counter on success
                    update_epics_pvs.connection_attempts[pv_name] = 0
                else:
                    logger.warning(f"PV {pv_name} not connected (attempt {update_epics_pvs.connection_attempts[pv_name]})")
            except Exception as e:
                logger.error(f"Error updating PV {pv_name}: {str(e)}")
                update_epics_pvs.connection_attempts[pv_name] += 1
    
    logger.info(f"Updated {updated_count} of {len(data)} EPICS PVs")
    
    # Track consecutive failures to detect persistent connection issues
    if updated_count == 0 and len(data) > 0:
        update_epics_pvs.consecutive_failures += 1
    else:
        update_epics_pvs.consecutive_failures = 0


if __name__ == "__main__":
    main()
