"""
STEP 1: Enhanced PDF Processing with Better Metadata Extraction
Improved citation support with page numbers and document structure
"""

import fitz  # PyMuPDF
import json
import re
from pathlib import Path
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

class EnhancedPDFProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        """
        Enhanced processor with better overlap for citation context
        """
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="text-embedding-3-large",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,  # Increased for better context
        )
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def extract_text_with_pages(self, pdf_path):
        """Extract text from PDF with page tracking"""
        doc = None
        try:
            doc = fitz.open(pdf_path)
            pages_data = []
            full_text = ""
            
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                page_start = len(full_text)
                full_text += page_text + f"\n[PAGE_MARKER_{page_num + 1}]\n"
                page_end = len(full_text)
                
                pages_data.append({
                    'page_num': page_num + 1,
                    'start_pos': page_start,
                    'end_pos': page_end,
                    'text': page_text
                })
            
            return full_text, pages_data, len(doc)
        except Exception as e:
            print(f"❌ Error processing {pdf_path.name}: {e}")
            return None, [], 0
        finally:
            if doc:
                doc.close()

    def extract_document_title(self, text, filename):
        """Extract document title from first few lines or filename"""
        lines = text.split('\n')[:10]
        for line in lines:
            line = line.strip()
            # Look for title-like patterns (longer lines near the start)
            if len(line) > 20 and len(line) < 200:
                # Avoid lines that are just numbers or dates
                if not re.match(r'^[\d\s\-/]+$', line):
                    return line
        
        # Fallback to cleaned filename
        title = filename.replace('.pdf', '').replace('-', ' ').replace('_', ' ')
        return title.title()

    def detect_legal_sections(self, text):
        """Enhanced section detection with more patterns"""
        patterns = [
            # English patterns
            (r'Section\s+(\d+[A-Za-z]?)[:\.\s]', 'section'),
            (r'Article\s+(\d+[A-Za-z]?)[:\.\s]', 'article'),
            (r'Chapter\s+(\d+|[IVXLCDM]+)[:\.\s]', 'chapter'),
            (r'Clause\s+(\d+[A-Za-z]?)[:\.\s]', 'clause'),
            (r'Rule\s+(\d+[A-Za-z]?)[:\.\s]', 'rule'),
            (r'Part\s+(\d+|[IVXLCDM]+)[:\.\s]', 'part'),
            
            # Nepali patterns
            (r'धारा\s+(\d+)', 'section_nepali'),
            (r'अनुच्छेद\s+(\d+)', 'article_nepali'),
            (r'परिच्छेद\s+(\d+)', 'chapter_nepali'),
            (r'नियम\s+(\d+)', 'rule_nepali'),
        ]
        
        sections = []
        for pattern, section_type in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
                sections.append({
                    'type': section_type,
                    'number': match.group(1),
                    'position': match.start(),
                    'text': match.group(0)
                })
        
        return sorted(sections, key=lambda x: x['position'])

    def get_pages_for_chunk(self, chunk_start, chunk_end, pages_data):
        """Determine which pages a chunk spans"""
        pages = []
        for page in pages_data:
            # Check if chunk overlaps with this page
            if not (chunk_end < page['start_pos'] or chunk_start > page['end_pos']):
                pages.append(page['page_num'])
        return pages

    def create_citation_string(self, metadata):
        """Create a formatted citation string"""
        parts = []
        
        if metadata.get('document_title'):
            parts.append(metadata['document_title'])
        
        if metadata.get('section_type') and metadata.get('section_number'):
            section_name = metadata['section_type'].replace('_nepali', '').title()
            parts.append(f"{section_name} {metadata['section_number']}")
        
        if metadata.get('page_numbers') and len(metadata['page_numbers']) > 0:
            pages = metadata['page_numbers']
            if len(pages) == 1:
                parts.append(f"p. {pages[0]}")
            else:
                parts.append(f"pp. {min(pages)}-{max(pages)}")
        
        return ", ".join(parts) if parts else metadata.get('filename', 'Unknown')

    def create_chunks(self, text, pages_data, metadata):
        """
        Enhanced chunking with page tracking and better metadata
        """
        sections = self.detect_legal_sections(text)
        all_chunks = []
        
        # Extract document title
        doc_title = self.extract_document_title(text, metadata['filename'])
        metadata['document_title'] = doc_title
        
        def process_text_segment(segment_text, start_pos, section_metadata=None):
            """Helper to process a text segment"""
            split_texts = self.text_splitter.split_text(segment_text)
            
            for chunk_text in split_texts:
                # Find position of chunk in original text
                chunk_start = text.find(chunk_text, start_pos)
                chunk_end = chunk_start + len(chunk_text)
                
                # Get pages for this chunk
                chunk_pages = self.get_pages_for_chunk(chunk_start, chunk_end, pages_data)
                
                # Build metadata
                chunk_metadata = {**metadata, 'chunk_index': len(all_chunks)}
                if section_metadata:
                    chunk_metadata.update(section_metadata)
                
                chunk_metadata['page_numbers'] = chunk_pages
                chunk_metadata['char_start'] = chunk_start
                chunk_metadata['char_end'] = chunk_end
                
                # Create citation
                chunk_metadata['citation'] = self.create_citation_string(chunk_metadata)
                
                all_chunks.append({
                    'text': chunk_text,
                    'metadata': chunk_metadata
                })
        
        last_pos = 0
        
        # Handle text before first section
        if sections:
            first_section_start = sections[0]['position']
            if first_section_start > 0:
                initial_text = text[:first_section_start].strip()
                if initial_text:
                    process_text_segment(initial_text, 0)
            last_pos = first_section_start

        # Process each section
        for i, section in enumerate(sections):
            start = section['position']
            end = sections[i + 1]['position'] if i + 1 < len(sections) else len(text)
            section_text = text[start:end].strip()

            section_metadata = {
                'section_type': section['type'],
                'section_number': section['number'],
                'section_title': section['text']
            }
            
            process_text_segment(section_text, start, section_metadata)

        # If no sections found, process entire document
        if not sections:
            process_text_segment(text, 0)

        return all_chunks

    def process_volume(self, volume_folder):
        """Process all PDFs in a volume folder with enhanced metadata"""
        volume_folder = Path(volume_folder)
        print(f"\n📂 Processing: {volume_folder.name}")
        pdf_files = list(volume_folder.glob("*.pdf"))
        print(f"   Found {len(pdf_files)} PDF files")

        all_chunks = []
        failed_files = []

        for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
            text, pages_data, num_pages = self.extract_text_with_pages(pdf_file)
            if not text:
                failed_files.append(pdf_file.name)
                continue

            metadata = {
                'filename': pdf_file.name,
                'volume': volume_folder.name,
                'total_pages': num_pages,
                'source': 'law_commission'
            }
            
            chunks = self.create_chunks(text, pages_data, metadata)
            all_chunks.extend(chunks)

        print(f"   ✓ Created {len(all_chunks)} chunks")
        if failed_files:
            print(f"   ⚠ Failed: {len(failed_files)} files")

        return all_chunks, failed_files


def main():
    print("=" * 70)
    print("   STEP 1: ENHANCED PDF PROCESSING")
    print("=" * 70)

    # Initialize with better overlap for citations
    processor = EnhancedPDFProcessor(chunk_size=1000, chunk_overlap=200)

    data_folder = Path("data")
    volume_folders = sorted(data_folder.glob("volume_*"))
    if not volume_folders:
        exclude_folders = {'processed', 'embeddings', '__pycache__'}
        volume_folders = sorted([d for d in data_folder.iterdir() 
                                if d.is_dir() and d.name not in exclude_folders])

    if not volume_folders:
        print(" No volume folders found in 'data/' directory.")
        return

    print(f"\nFound {len(volume_folders)} volume(s):")
    for vol in volume_folders:
        print(f"  - {vol.name}")

    all_chunks = []
    all_failed = []
    
    for volume_folder in volume_folders:
        chunks, failed = processor.process_volume(volume_folder)
        all_chunks.extend(chunks)
        all_failed.extend(failed)

    output_dir = Path("data/processed")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "all_chunks_enhanced.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    # Calculate statistics
    total_tokens = sum(len(processor.encoding.encode(c['text'])) for c in all_chunks)
    avg_chunk_size = total_tokens / len(all_chunks) if all_chunks else 0
    
    # Analyze metadata completeness
    chunks_with_sections = sum(1 for c in all_chunks if c['metadata'].get('section_number'))
    chunks_with_pages = sum(1 for c in all_chunks if c['metadata'].get('page_numbers'))
    chunks_with_titles = sum(1 for c in all_chunks if c['metadata'].get('document_title'))

    stats = {
        'total_chunks': len(all_chunks),
        'total_failed_files': len(all_failed),
        'failed_files': all_failed,
        'volumes_processed': [v.name for v in volume_folders],
        'avg_chunk_size_tokens': avg_chunk_size,
        'chunks_with_section_info': chunks_with_sections,
        'chunks_with_page_numbers': chunks_with_pages,
        'chunks_with_document_titles': chunks_with_titles,
        'metadata_completeness': {
            'sections': f"{chunks_with_sections/len(all_chunks)*100:.1f}%" if all_chunks else "0%",
            'pages': f"{chunks_with_pages/len(all_chunks)*100:.1f}%" if all_chunks else "0%",
            'titles': f"{chunks_with_titles/len(all_chunks)*100:.1f}%" if all_chunks else "0%"
        }
    }
    
    stats_file = output_dir / "processing_stats_enhanced.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("   PROCESSING COMPLETE!")
    print("=" * 70)
    print(f"✓ Total chunks created: {len(all_chunks)}")
    print(f"✓ Average chunk size: {stats['avg_chunk_size_tokens']:.0f} tokens")
    print(f"✓ Chunks with sections: {stats['chunks_with_section_info']} ({stats['metadata_completeness']['sections']})")
    print(f"✓ Chunks with pages: {stats['chunks_with_page_numbers']} ({stats['metadata_completeness']['pages']})")
    print(f"✓ Chunks with titles: {stats['chunks_with_titles']} ({stats['metadata_completeness']['titles']})")
    print(f"✓ Output saved to: {output_file}")
    print(f"✓ Stats saved to: {stats_file}")
    
    if all_failed:
        print(f"\n⚠ Warning: {len(all_failed)} files failed to process")
    
    print("\n🎯 Next step: Run step2_generate_embeddings.py")


if __name__ == "__main__":
    main()