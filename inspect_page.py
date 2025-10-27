from playwright.sync_api import sync_playwright
import time

def inspect_volume_page():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # Show browser
        page = browser.new_page()
        
        # Go to first volume page (update URL as needed)
        page.goto("https://lawcommission.gov.np/pages/list-volume-act/", wait_until='networkidle')
        
        # Click first volume link
        time.sleep(2)
        first_volume = page.locator('table tr td:nth-child(2) a').first
        first_volume.click()
        
        time.sleep(3)
        
        # Inspect table structure
        print("=== Inspecting Table Structure ===\n")
        
        rows = page.locator('table tbody tr').all()
        print(f"Found {len(rows)} rows\n")
        
        # Inspect first 3 rows
        for i in range(min(3, len(rows))):
            print(f"--- Row {i+1} ---")
            cells = rows[i].locator('td').all()
            print(f"Cells: {len(cells)}")
            
            for j, cell in enumerate(cells):
                text = cell.inner_text().strip()
                print(f"  Cell {j+1}: {text[:100]}")
                
                # Check for PDF link
                pdf = cell.locator('a[href$=".pdf"]')
                if pdf.count() > 0:
                    href = pdf.first.get_attribute('href')
                    print(f"    PDF: {href}")
            print()
        
        input("Press Enter to close browser...")
        browser.close()

if __name__ == "__main__":
    inspect_volume_page()
