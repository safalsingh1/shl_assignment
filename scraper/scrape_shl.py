import asyncio
import pandas as pd
import random
import time
from playwright.async_api import async_playwright
import os

INPUT_CSV = r"c:\Users\safal\Desktop\shl_assignment\shl_catalogue.csv"
OUTPUT_DIR = r"c:\Users\safal\Desktop\shl_assignment\data\raw"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "shl_catalogue.csv")

async def scrape_description(page, url):
    try:
        print(f"Visiting: {url}")
        await page.goto(url, timeout=60000, wait_until="domcontentloaded")
        
        await asyncio.sleep(1) 
        
        description = ""
        
        # Selectors to try for description text
        # These are educated guesses based on common layouts. 
        # browser_subagent showed a "Description" header.
        content_selectors = [
            ".product-description", 
            ".catalog-product-view__description", 
            "div[itemprop='description']",
            ".product-detail__description"
        ]
        
        for selector in content_selectors:
            if await page.locator(selector).count() > 0:
                description = await page.locator(selector).inner_text()
                break
        
        if not description:
            try:
                # Try finding a header with "Description" and getting the next sibling or parent text
                # We use a regex to be case-insensitive and handle potential whitespace
                # Look for h2, h3, h4, or strong/b tags that might be headers
                potential_headers = page.locator("h1, h2, h3, h4, h5, h6, strong, b").filter(has_text=pd.Series(["Description"]).str.contains("Description", case=False, regex=False).any() if False else "Description") 
                # Playwright filter is simpler:
                potential_headers = page.locator("h2, h3, h4, strong").filter(has_text="Description")
                
                count = await potential_headers.count()
                for i in range(count):
                    header = potential_headers.nth(i)
                    if await header.inner_text() == "Description":
                        # Attempt 1: Next sibling
                        next_sibling = header.locator("xpath=following-sibling::*[1]")
                        if await next_sibling.count() > 0:
                            text = await next_sibling.inner_text()
                            if len(text.strip()) > 10:
                                description = text
                                break
                        
                        # Attempt 2: Parent text
                        parent = header.locator("xpath=..")
                        text = await parent.inner_text()
                        cleaned = text.replace("Description", "", 1).strip()
                        if len(cleaned) > 10:
                            description = cleaned
                            break
            except Exception as e:
                pass


        if not description:
             # Last ditch: grabbed specific div from previous observation if possible?
             # Just return empty string if fail.
             pass

        return description.strip()

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return ""

async def main():
    # Resume from output if exists, else start from input
    if os.path.exists(OUTPUT_CSV):
        print(f"Resuming from {OUTPUT_CSV}")
        df = pd.read_csv(OUTPUT_CSV)
    elif os.path.exists(INPUT_CSV):
        print(f"Starting fresh from {INPUT_CSV}")
        df = pd.read_csv(INPUT_CSV)
    else:
        print(f"Input file not found: {INPUT_CSV}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Loaded {len(df)} rows")
    
    # Ensure description column exists and is string type
    if 'description' not in df.columns:
        df['description'] = ""
    
    # Fill NaN with empty string to avoid dtype issues
    df['description'] = df['description'].fillna("").astype(str)


    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )
        page = await context.new_page()

        for index, row in df.iterrows():
            url = row.get('assessment_url')
            if pd.isna(url) or not str(url).startswith('http'):
                continue
            
            # Resume capability
            if pd.notna(row.get('description')) and str(row.get('description')).strip() != "":
                continue

            desc = await scrape_description(page, url)
            
            # Clean up description (remove 'Description' header text if it was captured)
            if desc.startswith("Description"):
                desc = desc[11:].strip()
            
            df.at[index, 'description'] = desc
            
            if index % 10 == 0:
                df.to_csv(OUTPUT_CSV, index=False)
                print(f"Progress saved at index {index}")
            
            val = random.uniform(1.0, 2.0)
            await asyncio.sleep(val)

        df.to_csv(OUTPUT_CSV, index=False)
        print(f"Completed. Saved to {OUTPUT_CSV}")
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
