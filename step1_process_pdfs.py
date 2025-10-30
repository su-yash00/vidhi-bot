"""
STEP 2: Process PDFs and Create Clean Chunks
- Extracts text from PDFs
- Cleans text (removes OCR artifacts, page markers, extra whitespace)
- Detects sections (Nepali + English legal sections)
- Splits into token-based chunks using langchain RecursiveCharacterTextSplitter
- Saves cleaned chunks to JSON for embedding
"""

import fitz  # PyMuPDF
import json
import re
from pathlib import Path
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Metadata from your scraper
METADATA_FILE = Path("data/scraping_metadata.json")

class PDFChunker:
    def __init__(self, chunk_size=1000, chunk_overlap=150):
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="text-embedding-3-large",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def extract_text(self, pdf_path: Path):
        """Extract text from PDF"""
        doc = None
        try:
            doc = fitz.open(pdf_path)
            full_text = ""
            for i, page in enumerate(doc):
                full_text += page.get_text() + f"\n[PAGE {i + 1}]\n"
            return full_text, len(doc)
        except Exception as e:
            print(f"Error reading {pdf_path.name}: {e}")
            return None, 0
        finally:
            if doc:
                doc.close()

    def clean_text(self, text: str):
        """Clean OCR artifacts, page markers, extra whitespace"""
        text = re.sub(r'\[PAGE \d+\]', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\x00-\x7F\u0900-\u097F\s,.!?-]', '', text)  # Keep Nepali + ASCII
        return text.strip()

    def detect_sections(self, text: str):
        """Detect legal sections"""
        patterns = [
            (r'Section\s+(\d+[A-Za-z]?)[:\.\s]', 'section'),
            (r'Article\s+(\d+[A-Za-z]?)[:\.\s]', 'article'),
            (r'Chapter\s+(\d+|[IVXLCDM]+)[:\.\s]', 'chapter'),
            (r'धारा\s+(\d+)', 'section_nepali'),
            (r'अनुच्छेद\s+(\d+)', 'article_nepali'),
        ]
        sections = []
        for pattern, section_type in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
                sections.append({'type': section_type, 'number': match.group(1), 'position': match.start()})
        return sorted(sections, key=lambda x: x['position'])

    def chunk_text(self, text: str, metadata: dict):
        """Split text into chunks using sections + token splitter"""
        text = self.clean_text(text)
        sections = self.detect_sections(text)
        chunks = []

        # Before first section
        if sections:
            start = sections[0]['position']
            if start > 0:
                pre_text = text[:start].strip()
                if pre_text:
                    for c in self.text_splitter.split_text(pre_text):
                        chunks.append({'text': c, 'metadata': {**metadata, 'chunk_index': len(chunks)}})

        # Each section
        for i, sec in enumerate(sections):
            start = sec['position']
            end = sections[i + 1]['position'] if i + 1 < len(sections) else len(text)
            sec_text = text[start:end].strip()
            sec_meta = {**metadata, 'section_type': sec['type'], 'section_number': sec['number']}
            for c in self.text_splitter.split_text(sec_text):
                chunks.append({'text': c, 'metadata': {**sec_meta, 'chunk_index': len(chunks)}})

        # If no sections found, split entire text
        if not sections:
            for i, c in enumerate(self.text_splitter.split_text(text)):
                chunks.append({'text': c, 'metadata': {**metadata, 'chunk_index': i}})

        return chunks

    def process_folder(self, folder_path: Path):
        """Process all PDFs in a folder"""
        folder_path = Path(folder_path)
        if not folder_path.exists():
            print(f"⚠ Folder does not exist: {folder_path}")
            return [], []

        pdfs = [f for f in folder_path.iterdir() if f.suffix.lower() == '.pdf']
        all_chunks, failed = [], []

        for pdf in tqdm(pdfs, desc=f"Processing PDFs in {folder_path.name}"):
            text, num_pages = self.extract_text(pdf)
            if not text:
                failed.append(pdf.name)
                continue
            metadata = {'filename': pdf.name, 'volume': folder_path.name, 'total_pages': num_pages, 'source': 'law_commission'}
            chunks = self.chunk_text(text, metadata)
            all_chunks.extend(chunks)

        return all_chunks, failed

def load_scraping_metadata():
    if METADATA_FILE.exists():
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def main():
    print("="*70)
    print("STEP 2: PROCESS PDFs → CLEAN CHUNKS")
    print("="*70)

    processor = PDFChunker()
    metadata = load_scraping_metadata()

    all_chunks, all_failed = [], []

    for vol_name, vol_data in metadata.get('volumes', {}).items():
        folder_path = vol_data.get('folder_path', f"data/{vol_name}")
        chunks, failed = processor.process_folder(folder_path)
        all_chunks.extend(chunks)
        all_failed.extend(failed)

    output_dir = Path("data/processed")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "all_chunks_cleaned.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Total chunks created: {len(all_chunks)}")
    if all_failed:
        print(f"⚠ Warning: {len(all_failed)} files failed")
    print(f"Output saved to: {output_file}")
    print("\nNext: Run step3_generate_embeddings.py using this cleaned chunk file.")

if __name__ == "__main__":
    main()
