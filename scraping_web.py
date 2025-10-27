# import os
# import re
# import time
# import requests
# from bs4 import BeautifulSoup
# from urllib.parse import urljoin, unquote 
# from playwright.sync_api import sync_playwright

# # --- CONFIGURATION ---
# BASE_URL = "https://lawcommission.gov.np/"
# VOLUMES_LIST_URL = "https://lawcommission.gov.np/pages/list-volume-act/"
# DATA_FOLDER = "data"
# # ---------------------
  
# def get_volume_links(page_url):
#     print(f"Finding Volume links on: {page_url}")
#     try:
#         response = requests.get(page_url)
#         response.raise_for_status()
#         soup = BeautifulSoup(response.content, "html.parser")
#         volume_links = []
#         selector = "tr td:nth-child(2) a" 
#         for link in soup.select(selector):
#             href = link.get('href')
#             if href:
#                 full_url = urljoin(BASE_URL, href)
#                 volume_links.append(full_url)
#         print(f"Found {len(volume_links)} Volume links.")
#         return volume_links
#     except requests.exceptions.RequestException as e:
#         print(f"Error fetching the volumes list page: {e}")
#         return []

# def scrape_pdfs_with_pagination(volume_url):
#     print(f"\n--- Processing Volume with Pagination: {volume_url} ---")
#     all_pdf_links = {} 

#     with sync_playwright() as p:
#         browser = p.chromium.launch()
#         page = browser.new_page()
#         page.goto(volume_url, wait_until='networkidle')
        
#         page_num = 1
#         while True:
#             print(f"  Scraping page {page_num}...")
            
#             # Wait for at least one PDF link to be visible before scraping
#             page.wait_for_selector('a[href$=".pdf"]', timeout=15000)
            
#             links = page.locator('a[href$=".pdf"]').all()
            
#             for link in links:
#                 href = link.get_attribute('href')
#                 raw_filename = href.split('/')[-1]
#                 decoded_filename = unquote(raw_filename)
#                 name_without_ext = os.path.splitext(decoded_filename)[0]
#                 clean_name = re.sub(r'[\\/*?:"<>|\s]+', '-', name_without_ext)
#                 clean_name = clean_name[:100]
#                 file_name = f"{clean_name}.pdf"
#                 full_url = urljoin(BASE_URL, href)
#                 all_pdf_links[file_name] = full_url

#             # --- THE FINAL FIX BASED ON YOUR HTML ANALYSIS ---
#             # This selector perfectly matches the classes you found.
#             next_button_selector = 'a.pagination__btn.next__pagination'
#             next_button = page.locator(next_button_selector)
            
#             if next_button.count() > 0:
#                 print("  Found 'Next' button, clicking...")
#                 next_button.click()
#                 page.wait_for_load_state('networkidle')
#                 page_num += 1
#             else:
#                 print("  No more 'Next' button found. Finished this volume.")
#                 break
        
#         browser.close()

#     return all_pdf_links

# def download_files(pdf_links):
#     if not pdf_links:
#         return
    
#     print(f"\n--- Starting download of {len(pdf_links)} total PDFs ---")
#     for file_name, pdf_url in pdf_links.items():
#         full_path = os.path.join(DATA_FOLDER, file_name)
        
#         if os.path.exists(full_path):
#             print(f"Skipping '{file_name}', already exists.")
#             continue
        
#         try:
#             print(f"Downloading '{file_name}'...")
#             pdf_response = requests.get(pdf_url, stream=True, timeout=30)
#             pdf_response.raise_for_status()
#             with open(full_path, 'wb') as f:
#                 for chunk in pdf_response.iter_content(chunk_size=8192):
#                     f.write(chunk)
#             time.sleep(1)
#         except requests.exceptions.RequestException as e:
#             print(f"  -> Failed to download {pdf_url}. Error: {e}")

# def main():
#     print("--- Starting Definitive PDF Crawler (V8 - The Final Selector) ---")
#     if not os.path.exists(DATA_FOLDER):
#         os.makedirs(DATA_FOLDER)
    
#     volume_urls = get_volume_links(VOLUMES_LIST_URL)

#     if not volume_urls:
#         print("Could not find any Volume links. Exiting.")
#         return

#     # To test with just Volume 10 (at index 9)
#     target_volume_index = 9
#     if len(volume_urls) > target_volume_index:
#         url_to_scrape = volume_urls[target_volume_index]
#         links_from_volume = scrape_pdfs_with_pagination(url_to_scrape)
#         download_files(links_from_volume)
#     else:
#         print(f"Error: Could not find Volume 10 (index {target_volume_index}).")

#     print("\n--- Crawler Finished ---")
#     print("You are now ready to run ingest.py")

# if __name__ == "__main__":
#     main()

import os
import re
import time
import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, unquote 
from playwright.sync_api import sync_playwright
from datetime import datetime

# --- CONFIGURATION ---
BASE_URL = "https://lawcommission.gov.np/"
VOLUMES_LIST_URL = "https://lawcommission.gov.np/pages/list-volume-act/"
DATA_FOLDER = "data"
METADATA_FILE = "data/scraping_metadata.json"
# ---------------------

def load_metadata():
    """Load existing scraping metadata"""
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"volumes": {}, "last_updated": None}

def save_metadata(metadata):
    """Save scraping metadata"""
    metadata["last_updated"] = datetime.now().isoformat()
    os.makedirs(os.path.dirname(METADATA_FILE), exist_ok=True)
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

def get_volume_links(page_url):
    """Extract all volume links from the volumes list page"""
    print(f"Finding Volume links on: {page_url}")
    try:
        response = requests.get(page_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        
        volume_links = []
        selector = "tr td:nth-child(2) a" 
        
        for link in soup.select(selector):
            href = link.get('href')
            title = link.get_text(strip=True)
            if href:
                full_url = urljoin(BASE_URL, href)
                volume_links.append({
                    'url': full_url,
                    'title': title
                })
        
        print(f"Found {len(volume_links)} Volume links.")
        return volume_links
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the volumes list page: {e}")
        return []

def extract_volume_number(title_or_url):
    """Extract volume number from title or URL"""
    # Try to find "Volume X" or "भाग X" pattern
    match = re.search(r'(?:Volume|भाग)\s*(\d+)', title_or_url, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    # Try to find just a number
    match = re.search(r'volume-(\d+)', title_or_url, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    return None

def create_volume_folder(volume_info, base_folder=DATA_FOLDER):
    """Create a folder for a specific volume"""
    volume_num = extract_volume_number(volume_info['title']) or extract_volume_number(volume_info['url'])
    
    if volume_num:
        folder_name = f"volume_{volume_num:02d}"
    else:
        # Fallback: create folder from sanitized title
        safe_title = re.sub(r'[\\/*?:"<>|\s]+', '_', volume_info['title'])[:50]
        folder_name = safe_title or "volume_unknown"
    
    volume_folder = os.path.join(base_folder, folder_name)
    os.makedirs(volume_folder, exist_ok=True)
    
    return volume_folder, folder_name

def scrape_pdfs_with_pagination(volume_info, volume_folder):
    """Scrape all PDF links from a volume with pagination support"""
    volume_url = volume_info['url']
    print(f"\n--- Processing Volume: {volume_info['title']} ---")
    print(f"   URL: {volume_url}")
    print(f"   Saving to: {volume_folder}")
    
    all_pdf_links = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        try:
            page.goto(volume_url, wait_until='networkidle', timeout=60000)
            
            page_num = 1
            while True:
                print(f"  Scraping page {page_num}...")
                
                # Wait for PDF links
                try:
                    page.wait_for_selector('a[href$=".pdf"]', timeout=15000)
                except:
                    print(f"  No PDF links found on page {page_num}")
                    break
                
                # Extract all PDF links on current page
                links = page.locator('a[href$=".pdf"]').all()
                print(f"  Found {len(links)} PDF links on page {page_num}")
                
                for link in links:
                    href = link.get_attribute('href')
                    if not href:
                        continue
                    
                    # Extract and clean filename
                    raw_filename = href.split('/')[-1]
                    decoded_filename = unquote(raw_filename)
                    name_without_ext = os.path.splitext(decoded_filename)[0]
                    clean_name = re.sub(r'[\\/*?:"<>|\s]+', '-', name_without_ext)
                    clean_name = clean_name[:100]
                    file_name = f"{clean_name}.pdf"
                    
                    # Get full URL
                    full_url = urljoin(BASE_URL, href)
                    
                    # Store with metadata
                    all_pdf_links[file_name] = {
                        'url': full_url,
                        'page_number': page_num,
                        'original_filename': decoded_filename
                    }
                
                # Check for next button
                next_button_selector = 'a.pagination__btn.next__pagination'
                next_button = page.locator(next_button_selector)
                
                if next_button.count() > 0:
                    print("  Found 'Next' button, clicking...")
                    next_button.click()
                    page.wait_for_load_state('networkidle', timeout=30000)
                    page_num += 1
                    time.sleep(1)  # Small delay between pages
                else:
                    print("  No more 'Next' button found. Finished this volume.")
                    break
        
        except Exception as e:
            print(f"  Error during scraping: {e}")
        finally:
            browser.close()

    print(f"  Total PDFs found in this volume: {len(all_pdf_links)}")
    return all_pdf_links

def download_files(pdf_links, volume_folder, volume_name):
    """Download all PDFs to the specified volume folder"""
    if not pdf_links:
        print(f"No PDFs to download for {volume_name}")
        return
    
    print(f"\n--- Downloading {len(pdf_links)} PDFs to {volume_name} ---")
    
    downloaded = 0
    skipped = 0
    failed = 0
    
    for file_name, pdf_info in pdf_links.items():
        pdf_url = pdf_info['url']
        full_path = os.path.join(volume_folder, file_name)
        
        if os.path.exists(full_path):
            print(f"  ✓ Skipping '{file_name}', already exists.")
            skipped += 1
            continue
        
        try:
            print(f"  ⬇ Downloading '{file_name}'...")
            pdf_response = requests.get(pdf_url, stream=True, timeout=30)
            pdf_response.raise_for_status()
            
            with open(full_path, 'wb') as f:
                for chunk in pdf_response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            downloaded += 1
            print(f"    ✓ Success")
            time.sleep(1)  # Rate limiting
            
        except requests.exceptions.RequestException as e:
            print(f"    ✗ Failed to download. Error: {e}")
            failed += 1
    
    print(f"\n--- Download Summary for {volume_name} ---")
    print(f"  Downloaded: {downloaded}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed: {failed}")
    
    return downloaded, skipped, failed

def process_single_volume(volume_info, metadata):
    """Process a single volume: scrape and download"""
    # Create volume folder
    volume_folder, folder_name = create_volume_folder(volume_info)
    
    # Scrape PDF links
    pdf_links = scrape_pdfs_with_pagination(volume_info, volume_folder)
    
    # Download PDFs
    downloaded, skipped, failed = download_files(pdf_links, volume_folder, folder_name)
    
    # Update metadata
    metadata['volumes'][folder_name] = {
        'title': volume_info['title'],
        'url': volume_info['url'],
        'folder': volume_folder,
        'total_pdfs': len(pdf_links),
        'downloaded': downloaded,
        'skipped': skipped,
        'failed': failed,
        'last_scraped': datetime.now().isoformat()
    }
    
    return metadata

def main():
    print("=" * 70)
    print("   NEPALI LEGAL RAG - VOLUME-BASED PDF SCRAPER")
    print("=" * 70)
    
    # Create base data folder
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
    
    # Load existing metadata
    metadata = load_metadata()
    
    # Get all volume links
    volume_links = get_volume_links(VOLUMES_LIST_URL)
    
    if not volume_links:
        print("❌ Could not find any Volume links. Exiting.")
        return
    
    print(f"\nFound {len(volume_links)} volumes total")
    for i, vol in enumerate(volume_links):
        print(f"  [{i}] {vol['title']}")
    
    # --- CONFIGURATION: Which volumes to process ---
    # Option 1: Process a specific volume by index
    PROCESS_SPECIFIC_VOLUMES = True  # Set to False to process all
    TARGET_VOLUME_INDICES = [9]  # Volume 10 (0-indexed)
    
    # Option 2: Process a range of volumes
    # TARGET_VOLUME_INDICES = range(0, 5)  # First 5 volumes
    
    # Option 3: Process all volumes
    # PROCESS_SPECIFIC_VOLUMES = False
    # -----------------------------------------------
    
    if PROCESS_SPECIFIC_VOLUMES:
        volumes_to_process = [volume_links[i] for i in TARGET_VOLUME_INDICES if i < len(volume_links)]
        print(f"\n📋 Processing {len(volumes_to_process)} specific volume(s)")
    else:
        volumes_to_process = volume_links
        print(f"\n📋 Processing ALL {len(volumes_to_process)} volumes")
    
    # Process each volume
    total_downloaded = 0
    total_failed = 0
    
    for i, volume_info in enumerate(volumes_to_process, 1):
        print(f"\n{'=' * 70}")
        print(f"  VOLUME {i}/{len(volumes_to_process)}")
        print(f"{'=' * 70}")
        
        try:
            metadata = process_single_volume(volume_info, metadata)
            
            # Save metadata after each volume
            save_metadata(metadata)
            
            volume_stats = list(metadata['volumes'].values())[-1]
            total_downloaded += volume_stats['downloaded']
            total_failed += volume_stats['failed']
            
        except Exception as e:
            print(f"❌ Error processing volume: {e}")
            continue
    
    # Final summary
    print(f"\n{'=' * 70}")
    print("   SCRAPING COMPLETE!")
    print(f"{'=' * 70}")
    print(f"Total volumes processed: {len(metadata['volumes'])}")
    print(f"Total PDFs downloaded: {total_downloaded}")
    print(f"Total failures: {total_failed}")
    print(f"\nMetadata saved to: {METADATA_FILE}")
    print(f"PDFs organized in: {DATA_FOLDER}/")
    print("\n✓ Ready to run next steps of the pipeline!")

if __name__ == "__main__":
    main()



# import os
# import re
# import time
# import json
# import requests
# from bs4 import BeautifulSoup
# from urllib.parse import urljoin, unquote 
# from playwright.sync_api import sync_playwright
# from datetime import datetime

# # --- CONFIGURATION ---
# BASE_URL = "https://lawcommission.gov.np/"
# VOLUMES_LIST_URL = "https://lawcommission.gov.np/pages/list-volume-act/"
# DATA_FOLDER = "data"
# METADATA_FILE = "data/scraping_metadata.json"
# # ---------------------

# def load_metadata():
#     """Load existing scraping metadata"""
#     if os.path.exists(METADATA_FILE):
#         with open(METADATA_FILE, 'r', encoding='utf-8') as f:
#             return json.load(f)
#     return {"volumes": {}, "last_updated": None}

# def save_metadata(metadata):
#     """Save scraping metadata"""
#     metadata["last_updated"] = datetime.now().isoformat()
#     os.makedirs(os.path.dirname(METADATA_FILE), exist_ok=True)
#     with open(METADATA_FILE, 'w', encoding='utf-8') as f:
#         json.dump(metadata, f, indent=2, ensure_ascii=False)

# def get_volume_links_with_playwright():
#     """
#     Get volume links using Playwright with English language selection
#     """
#     print(f"Getting volume links from: {VOLUMES_LIST_URL}")
#     volume_links = []
    
#     with sync_playwright() as p:
#         browser = p.chromium.launch(headless=False)  # Set to False to see what's happening
#         page = browser.new_page()
        
#         try:
#             # Go to volumes list page
#             page.goto(VOLUMES_LIST_URL, wait_until='networkidle', timeout=60000)
#             time.sleep(2)
            
#             # Try to find and click English language toggle
#             # Common selectors for language switchers
#             language_selectors = [
#                 'a:has-text("ENG")',
#                 'a:has-text("English")',
#                 'button:has-text("ENG")',
#                 'button:has-text("English")',
#                 '[data-lang="en"]',
#                 '[data-language="english"]',
#                 '.language-switcher a:has-text("ENG")',
#                 'a[href*="lang=en"]',
#             ]
            
#             switched = False
#             for selector in language_selectors:
#                 try:
#                     if page.locator(selector).count() > 0:
#                         print(f"Found language toggle: {selector}")
#                         page.locator(selector).first.click()
#                         time.sleep(2)
#                         switched = True
#                         print("✓ Switched to English")
#                         break
#                 except:
#                     continue
            
#             if not switched:
#                 print("⚠ Could not find English toggle, continuing with current language...")
            
#             # Wait for content to load
#             page.wait_for_load_state('networkidle')
#             time.sleep(2)
            
#             # Get all volume links from the table
#             # Try multiple selectors
#             selectors_to_try = [
#                 'table tr td:nth-child(2) a',
#                 'table tr td a',
#                 '.volume-list a',
#                 'tbody tr td:nth-child(2) a'
#             ]
            
#             links_found = []
#             for selector in selectors_to_try:
#                 try:
#                     elements = page.locator(selector).all()
#                     if elements:
#                         links_found = elements
#                         print(f"✓ Found links using selector: {selector}")
#                         break
#                 except:
#                     continue
            
#             if not links_found:
#                 print("❌ Could not find volume links with any selector")
#                 browser.close()
#                 return []
            
#             # Extract link information
#             for idx, link in enumerate(links_found):
#                 try:
#                     href = link.get_attribute('href')
#                     text = link.inner_text().strip()
                    
#                     if href and text:
#                         full_url = urljoin(BASE_URL, href)
                        
#                         # Try to extract volume number from text
#                         volume_num = extract_volume_number_from_text(text, idx)
                        
#                         volume_links.append({
#                             'url': full_url,
#                             'title': text,
#                             'volume_number': volume_num,
#                             'index': idx
#                         })
#                         print(f"  [{idx}] Volume {volume_num}: {text}")
#                 except Exception as e:
#                     print(f"  Error extracting link {idx}: {e}")
#                     continue
            
#         except Exception as e:
#             print(f"Error during scraping: {e}")
#         finally:
#             browser.close()
    
#     print(f"\n✓ Found {len(volume_links)} volume links")
#     return volume_links

# def extract_volume_number_from_text(text, fallback_index):
#     """
#     Extract volume number from text (works for both English and Nepali)
#     """
#     # Try English patterns
#     patterns = [
#         r'Volume[\s-]*(\d+)',
#         r'Vol[\s\.]*(\d+)',
#         r'V[\s\.]*(\d+)',
#         r'Part[\s-]*(\d+)',
#         r'Book[\s-]*(\d+)',
#     ]
    
#     for pattern in patterns:
#         match = re.search(pattern, text, re.IGNORECASE)
#         if match:
#             return int(match.group(1))
    
#     # Try Nepali numerals patterns
#     nepali_patterns = [
#         r'भाग[\s-]*(\d+)',
#         r'खण्ड[\s-]*(\d+)',
#     ]
    
#     for pattern in nepali_patterns:
#         match = re.search(pattern, text)
#         if match:
#             return int(match.group(1))
    
#     # Try to find any number in the text
#     numbers = re.findall(r'\d+', text)
#     if numbers:
#         return int(numbers[0])
    
#     # Fallback to index + 1
#     return fallback_index + 1

# def create_volume_folder(volume_info, base_folder=DATA_FOLDER):
#     """Create a folder for a specific volume using standardized naming"""
#     volume_num = volume_info.get('volume_number', volume_info.get('index', 0) + 1)
    
#     # Create standardized folder name: volume_01, volume_02, etc.
#     folder_name = f"volume_{volume_num:02d}"
#     volume_folder = os.path.join(base_folder, folder_name)
#     os.makedirs(volume_folder, exist_ok=True)
    
#     return volume_folder, folder_name


# # def scrape_pdfs_with_pagination(volume_info, volume_folder):
#     """Scrape all PDF links from a volume with pagination support"""
#     volume_url = volume_info['url']
#     volume_num = volume_info.get('volume_number', 'Unknown')
    
#     print(f"\n--- Processing Volume {volume_num}: {volume_info['title']} ---")
#     print(f"   URL: {volume_url}")
#     print(f"   Saving to: {volume_folder}")
    
#     all_pdf_links = {}

#     with sync_playwright() as p:
#         browser = p.chromium.launch(headless=True)
#         page = browser.new_page()
        
#         try:
#             page.goto(volume_url, wait_until='networkidle', timeout=60000)
            
#             page_num = 1
#             while True:
#                 print(f"  Scraping page {page_num}...")
                
#                 # Wait for PDF links
#                 try:
#                     page.wait_for_selector('a[href$=".pdf"]', timeout=15000)
#                 except:
#                     print(f"  No PDF links found on page {page_num}")
#                     break
                
#                 # Extract all PDF links on current page
#                 links = page.locator('a[href$=".pdf"]').all()
#                 print(f"  Found {len(links)} PDF links on page {page_num}")
                
#                 for link in links:
#                     href = link.get_attribute('href')
#                     if not href:
#                         continue
                    
#                     # Extract and clean filename
#                     raw_filename = href.split('/')[-1]
#                     decoded_filename = unquote(raw_filename)
#                     name_without_ext = os.path.splitext(decoded_filename)[0]
#                     clean_name = re.sub(r'[\\/*?:"<>|\s]+', '-', name_without_ext)
#                     clean_name = clean_name[:100]
#                     file_name = f"{clean_name}.pdf"
                    
#                     # Get full URL
#                     full_url = urljoin(BASE_URL, href)
                    
#                     # Store with metadata
#                     all_pdf_links[file_name] = {
#                         'url': full_url,
#                         'page_number': page_num,
#                         'original_filename': decoded_filename
#                     }
                
#                 # Check for next button
#                 next_button_selector = 'a.pagination__btn.next__pagination'
#                 next_button = page.locator(next_button_selector)
                
#                 if next_button.count() > 0:
#                     print("  Found 'Next' button, clicking...")
#                     next_button.click()
#                     page.wait_for_load_state('networkidle', timeout=30000)
#                     page_num += 1
#                     time.sleep(1)
#                 else:
#                     print("  No more 'Next' button found. Finished this volume.")
#                     break
        
#         except Exception as e:
#             print(f"  Error during scraping: {e}")
#         finally:
#             browser.close()

#     print(f"  Total PDFs found in this volume: {len(all_pdf_links)}")
#     return all_pdf_links

# # def scrape_pdfs_with_pagination(volume_info, volume_folder):
# #     """Scrape all PDF links from a volume with pagination support"""
# #     volume_url = volume_info['url']
# #     volume_num = volume_info.get('volume_number', 'Unknown')
    
# #     print(f"\n--- Processing Volume {volume_num}: {volume_info['title']} ---")
# #     print(f"   URL: {volume_url}")
# #     print(f"   Saving to: {volume_folder}")
    
# #     all_pdf_links = {}
# #     pdf_counter = 1  # Counter for sequential naming

# #     with sync_playwright() as p:
# #         browser = p.chromium.launch(headless=True)
# #         page = browser.new_page()
        
# #         try:
# #             page.goto(volume_url, wait_until='networkidle', timeout=60000)
            
# #             page_num = 1
# #             while True:
# #                 print(f"  Scraping page {page_num}...")
                
# #                 # Wait for PDF links
# #                 try:
# #                     page.wait_for_selector('a[href$=".pdf"]', timeout=15000)
# #                 except:
# #                     print(f"  No PDF links found on page {page_num}")
# #                     break
                
# #                 # Extract all PDF links on current page
# #                 links = page.locator('a[href$=".pdf"]').all()
# #                 print(f"  Found {len(links)} PDF links on page {page_num}")
                
# #                 for link in links:
# #                     href = link.get_attribute('href')
# #                     if not href:
# #                         continue
                    
# #                     # Get the link text (document title)
# #                     link_text = link.inner_text().strip()
                    
# #                     # Try to get title from parent element if link text is empty
# #                     if not link_text or len(link_text) < 5:
# #                         try:
# #                             # Try to find title in parent row
# #                             parent = link.locator('xpath=../..').first
# #                             link_text = parent.inner_text().strip()
# #                             # Clean up (remove extra whitespace and newlines)
# #                             link_text = ' '.join(link_text.split())
# #                         except:
# #                             pass
                    
# #                     # If still no good title, try aria-label or title attribute
# #                     if not link_text or len(link_text) < 5:
# #                         link_text = link.get_attribute('title') or link.get_attribute('aria-label') or ''
                    
# #                     # Fallback to URL filename if no title found
# #                     if not link_text or len(link_text) < 5:
# #                         url_filename = href.split('/')[-1]
# #                         link_text = unquote(url_filename)
                    
# #                     # Create clean filename from title
# #                     clean_name = create_clean_filename(link_text, pdf_counter, volume_num)
# #                     file_name = f"{clean_name}.pdf"
                    
# #                     # Get full URL
# #                     full_url = urljoin(BASE_URL, href)
                    
# #                     # Avoid duplicates
# #                     if file_name in all_pdf_links:
# #                         file_name = f"{clean_name}_{pdf_counter}.pdf"
                    
# #                     # Store with metadata
# #                     all_pdf_links[file_name] = {
# #                         'url': full_url,
# #                         'page_number': page_num,
# #                         'original_filename': href.split('/')[-1],
# #                         'title': link_text
# #                     }
                    
# #                     pdf_counter += 1
                
# #                 # Check for next button
# #                 next_button_selector = 'a.pagination__btn.next__pagination'
# #                 next_button = page.locator(next_button_selector)
                
# #                 if next_button.count() > 0:
# #                     print("  Found 'Next' button, clicking...")
# #                     next_button.click()
# #                     page.wait_for_load_state('networkidle', timeout=30000)
# #                     page_num += 1
# #                     time.sleep(1)
# #                 else:
# #                     print("  No more 'Next' button found. Finished this volume.")
# #                     break
        
# #         except Exception as e:
# #             print(f"  Error during scraping: {e}")
# #         finally:
# #             browser.close()

# #     print(f"  Total PDFs found in this volume: {len(all_pdf_links)}")
# #     return all_pdf_links

# def scrape_pdfs_with_pagination(volume_info, volume_folder):
#     """Scrape all PDF links from a volume using table structure"""
#     volume_url = volume_info['url']
#     volume_num = volume_info.get('volume_number', 'Unknown')
    
#     print(f"\n--- Processing Volume {volume_num}: {volume_info['title']} ---")
#     print(f"   URL: {volume_url}")
#     print(f"   Saving to: {volume_folder}")
    
#     all_pdf_links = {}
#     pdf_counter = 1

#     with sync_playwright() as p:
#         browser = p.chromium.launch(headless=True)
#         page = browser.new_page()
        
#         try:
#             page.goto(volume_url, wait_until='networkidle', timeout=60000)
            
#             page_num = 1
#             while True:
#                 print(f"  Scraping page {page_num}...")
                
#                 # Wait for table rows
#                 try:
#                     page.wait_for_selector('table tbody tr', timeout=15000)
#                 except:
#                     print(f"  No table rows found on page {page_num}")
#                     break
                
#                 # Get all table rows
#                 rows = page.locator('table tbody tr').all()
#                 print(f"  Found {len(rows)} rows on page {page_num}")
                
#                 for row in rows:
#                     try:
#                         # Get all cells in the row
#                         cells = row.locator('td').all()
                        
#                         if len(cells) < 2:
#                             continue
                        
#                         # First column usually has S.N., second has title, third/fourth has PDF
#                         title_text = ""
#                         pdf_link = None
#                         pdf_href = None
                        
#                         # Try to find title (usually in column 1 or 2)
#                         for i in range(min(3, len(cells))):
#                             text = cells[i].inner_text().strip()
#                             # Skip if it's just a number (S.N.)
#                             if text and not text.isdigit() and len(text) > 5:
#                                 title_text = text
#                                 break
                        
#                         # Find PDF link in any column
#                         for cell in cells:
#                             pdf_link = cell.locator('a[href$=".pdf"]').first
#                             if pdf_link.count() > 0:
#                                 pdf_href = pdf_link.get_attribute('href')
#                                 # If title not found yet, try link text
#                                 if not title_text:
#                                     title_text = pdf_link.inner_text().strip()
#                                 break
                        
#                         if not pdf_href:
#                             continue
                        
#                         # Fallback title
#                         if not title_text or len(title_text) < 5:
#                             title_text = f"Document {pdf_counter}"
                        
#                         # Create clean filename
#                         clean_name = create_clean_filename(title_text, pdf_counter, volume_num)
#                         file_name = f"{clean_name}.pdf"
                        
#                         # Get full URL
#                         full_url = urljoin(BASE_URL, pdf_href)
                        
#                         # Avoid duplicates
#                         if file_name in all_pdf_links:
#                             file_name = f"{clean_name}_{pdf_counter}.pdf"
                        
#                         # Store with metadata
#                         all_pdf_links[file_name] = {
#                             'url': full_url,
#                             'page_number': page_num,
#                             'original_filename': pdf_href.split('/')[-1],
#                             'title': title_text
#                         }
                        
#                         print(f"    [{pdf_counter}] {title_text[:60]}...")
#                         pdf_counter += 1
                        
#                     except Exception as e:
#                         print(f"    Error processing row: {e}")
#                         continue
                
#                 # Check for next button
#                 next_button_selector = 'a.pagination__btn.next__pagination'
#                 next_button = page.locator(next_button_selector)
                
#                 if next_button.count() > 0:
#                     print("  Found 'Next' button, clicking...")
#                     next_button.click()
#                     page.wait_for_load_state('networkidle', timeout=30000)
#                     page_num += 1
#                     time.sleep(1)
#                 else:
#                     print("  No more 'Next' button found. Finished this volume.")
#                     break
        
#         except Exception as e:
#             print(f"  Error during scraping: {e}")
#         finally:
#             browser.close()

#     print(f"  Total PDFs found in this volume: {len(all_pdf_links)}")
#     return all_pdf_links

# def create_clean_filename(title, counter, volume_num):
#     """
#     Create a clean, readable filename from document title
#     """
#     # Remove common prefixes/suffixes
#     title = re.sub(r'^(Act|Regulation|Rule|Code|Law|ऐन|नियम|कानून|विधान)\s*[-:]\s*', '', title, flags=re.IGNORECASE)
    
#     # Replace Nepali and special characters with safe alternatives
#     replacements = {
#         'ऐन': 'Ain',
#         'नियम': 'Niyam',
#         'कानून': 'Kanun',
#         'विधान': 'Vidhan',
#         'र': 'Ra',
#         'को': 'Ko',
#         'ले': 'Le',
#         'मा': 'Ma',
#         'सम्बन्धी': 'Sambandhi',
#         'तथा': 'Tatha',
#         '/': '-',
#         '\\': '-',
#         ':': '-',
#         '*': '',
#         '?': '',
#         '"': '',
#         '<': '',
#         '>': '',
#         '|': '-',
#         '\n': '-',
#         '\r': '',
#         '\t': '-',
#     }
    
#     for old, new in replacements.items():
#         title = title.replace(old, new)
    
#     # Remove multiple spaces and dashes
#     title = re.sub(r'\s+', '-', title)
#     title = re.sub(r'-+', '-', title)
    
#     # Remove leading/trailing dashes
#     title = title.strip('-')
    
#     # Limit length
#     if len(title) > 100:
#         title = title[:100]
    
#     # If title is too short or only special chars, use counter-based naming
#     if len(title) < 5 or not any(c.isalnum() for c in title):
#         title = f"vol{volume_num}_doc{counter:03d}"
    
#     return title

# def download_files(pdf_links, volume_folder, volume_name):
#     """Download all PDFs to the specified volume folder"""
#     if not pdf_links:
#         print(f"No PDFs to download for {volume_name}")
#         return 0, 0, 0
    
#     print(f"\n--- Downloading {len(pdf_links)} PDFs to {volume_name} ---")
    
#     downloaded = 0
#     skipped = 0
#     failed = 0
    
#     for file_name, pdf_info in pdf_links.items():
#         pdf_url = pdf_info['url']
#         full_path = os.path.join(volume_folder, file_name)
        
#         if os.path.exists(full_path):
#             print(f"  ✓ Skipping '{file_name}', already exists.")
#             skipped += 1
#             continue
        
#         try:
#             print(f"  ⬇ Downloading '{file_name}'...")
#             pdf_response = requests.get(pdf_url, stream=True, timeout=30)
#             pdf_response.raise_for_status()
            
#             with open(full_path, 'wb') as f:
#                 for chunk in pdf_response.iter_content(chunk_size=8192):
#                     f.write(chunk)
            
#             downloaded += 1
#             print(f"    ✓ Success")
#             time.sleep(1)
            
#         except requests.exceptions.RequestException as e:
#             print(f"    ✗ Failed to download. Error: {e}")
#             failed += 1
    
#     print(f"\n--- Download Summary for {volume_name} ---")
#     print(f"  Downloaded: {downloaded}")
#     print(f"  Skipped: {skipped}")
#     print(f"  Failed: {failed}")
    
#     return downloaded, skipped, failed

# def process_single_volume(volume_info, metadata):
#     """Process a single volume: scrape and download"""
#     # Create volume folder
#     volume_folder, folder_name = create_volume_folder(volume_info)
    
#     # Scrape PDF links
#     pdf_links = scrape_pdfs_with_pagination(volume_info, volume_folder)
    
#     # Download PDFs
#     downloaded, skipped, failed = download_files(pdf_links, volume_folder, folder_name)
    
#     # Update metadata
#     metadata['volumes'][folder_name] = {
#         'title': volume_info['title'],
#         'volume_number': volume_info.get('volume_number'),
#         'url': volume_info['url'],
#         'folder': volume_folder,
#         'total_pdfs': len(pdf_links),
#         'downloaded': downloaded,
#         'skipped': skipped,
#         'failed': failed,
#         'last_scraped': datetime.now().isoformat()
#     }
    
#     return metadata

# def main():
#     print("=" * 70)
#     print("   NEPALI LEGAL RAG - IMPROVED VOLUME SCRAPER")
#     print("=" * 70)
    
#     # Create base data folder
#     if not os.path.exists(DATA_FOLDER):
#         os.makedirs(DATA_FOLDER)
    
#     # Load existing metadata
#     metadata = load_metadata()
    
#     # Get all volume links with English language
#     volume_links = get_volume_links_with_playwright()
    
#     if not volume_links:
#         print("❌ Could not find any Volume links. Exiting.")
#         return
    
#     print(f"\nFound {len(volume_links)} volumes total")
    
#     # --- CONFIGURATION: Which volumes to process ---
#     PROCESS_SPECIFIC_VOLUMES = False  # Set to True to process specific volumes
#     TARGET_VOLUME_INDICES = [0]  # Example: process first volume only
    
#     # To process ALL volumes, keep PROCESS_SPECIFIC_VOLUMES = False
#     # To process specific volumes, set PROCESS_SPECIFIC_VOLUMES = True and set indices
#     # -----------------------------------------------
    
#     if PROCESS_SPECIFIC_VOLUMES:
#         volumes_to_process = [volume_links[i] for i in TARGET_VOLUME_INDICES if i < len(volume_links)]
#         print(f"\n📋 Processing {len(volumes_to_process)} specific volume(s)")
#     else:
#         volumes_to_process = volume_links
#         print(f"\n📋 Processing ALL {len(volumes_to_process)} volumes")
    
#     # Process each volume
#     total_downloaded = 0
#     total_failed = 0
    
#     for i, volume_info in enumerate(volumes_to_process, 1):
#         print(f"\n{'=' * 70}")
#         print(f"  VOLUME {i}/{len(volumes_to_process)}")
#         print(f"{'=' * 70}")
        
#         try:
#             metadata = process_single_volume(volume_info, metadata)
            
#             # Save metadata after each volume
#             save_metadata(metadata)
            
#             volume_stats = list(metadata['volumes'].values())[-1]
#             total_downloaded += volume_stats['downloaded']
#             total_failed += volume_stats['failed']
            
#         except Exception as e:
#             print(f"❌ Error processing volume: {e}")
#             continue
    
#     # Final summary
#     print(f"\n{'=' * 70}")
#     print("   SCRAPING COMPLETE!")
#     print(f"{'=' * 70}")
#     print(f"Total volumes processed: {len(metadata['volumes'])}")
#     print(f"Total PDFs downloaded: {total_downloaded}")
#     print(f"Total failures: {total_failed}")
#     print(f"\nMetadata saved to: {METADATA_FILE}")
#     print(f"PDFs organized in: {DATA_FOLDER}/volume_XX/")
#     print("\n✓ Ready to run next steps of the pipeline!")

# if __name__ == "__main__":
#     main()