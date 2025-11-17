from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time

URL = "https://example.com/login"  # replace with real login URL
VALID = {"username": "validuser", "password": "validpass"}
INVALID = {"username": "bad", "password": "bad"}

def run_test_case(driver, creds, timeout=5):
    driver.get(URL)
    time.sleep(1)
    # Update selectors to match page
    driver.find_element(By.NAME, "username").clear()
    driver.find_element(By.NAME, "username").send_keys(creds["username"])
    driver.find_element(By.NAME, "password").clear()
    driver.find_element(By.NAME, "password").send_keys(creds["password"])
    driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click()
    time.sleep(timeout)
    # Determine success/failure — adapt to app (URL change, element presence, message)
    try:
        # Example: successful login redirects to dashboard or contains logout link
        driver.find_element(By.ID, "logout")
        return True
    except:
        return False

def main():
    chrome_options = Options()
    # chrome_options.add_argument("--headless")  # uncomment to run headless
    driver = webdriver.Chrome(ChromeDriverManager().install(), options=chrome_options)
    try:
        runs = [
            (VALID, "valid"),
            (INVALID, "invalid")
        ]
        results = {"valid": [], "invalid": []}
        # Run each case multiple times to collect rates
        for creds, label in runs:
            for i in range(5):
                ok = run_test_case(driver, creds)
                results[label].append(ok)
                driver.save_screenshot(f"./screenshot_{label}_{i}.png")
        # Print summary
        for label in results:
            successes = sum(results[label])
            total = len(results[label])
            print(f"{label}: {successes}/{total} passed ({successes/total:.2%})")
    finally:
        driver.quit()

if __name__ == "__main__":
    main()