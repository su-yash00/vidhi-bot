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

