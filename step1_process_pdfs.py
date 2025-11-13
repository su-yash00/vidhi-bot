# """
# STEP 1: Process PDFs and Create Chunks (IMPROVED VERSION)
# This extracts text from PDFs and splits them into meaningful chunks
# using a robust text splitter to prevent oversized chunks.
# """

# import fitz  # PyMuPDF
# import json
# import re
# from pathlib import Path
# from tqdm import tqdm
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import tiktoken  # Added for accurate token counting

# class SimplePDFProcessor:
#     def __init__(self, chunk_size=1000, chunk_overlap=150):
#         """
#         Initializes the processor with a token-based text splitter.
#         chunk_size and chunk_overlap are now measured in TOKENS, not words.
#         """
#         self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#             model_name="text-embedding-3-large",  # Match the model in Step 2
#             chunk_size=chunk_size,
#             chunk_overlap=chunk_overlap,
#         )

#     def extract_text_from_pdf(self, pdf_path):
#         """Extract text from a single PDF"""
#         doc = None
#         try:
#             doc = fitz.open(pdf_path)
#             full_text = ""
#             for page_num, page in enumerate(doc):
#                 full_text += page.get_text() + f"\n[PAGE {page_num + 1}]\n"
#             return full_text, len(doc)
#         except Exception as e:
#             print(f"❌ Error processing {pdf_path.name}: {e}")
#             return None, 0
#         finally:
#             if doc:
#                 doc.close()

#     def detect_legal_sections(self, text):
#         """Detect legal document sections (your original, effective logic)"""
#         patterns = [
#             (r'Section\s+(\d+[A-Za-z]?)[:\.\s]', 'section'),
#             (r'Article\s+(\d+[A-Za-z]?)[:\.\s]', 'article'),
#             (r'Chapter\s+(\d+|[IVXLCDM]+)[:\.\s]', 'chapter'),
#             (r'धारा\s+(\d+)', 'section_nepali'),
#             (r'अनुच्छेद\s+(\d+)', 'article_nepali'),
#         ]
#         sections = []
#         for pattern, section_type in patterns:
#             for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
#                 sections.append({
#                     'type': section_type,
#                     'number': match.group(1),
#                     'position': match.start()
#                 })
#         return sorted(sections, key=lambda x: x['position'])

#     def create_chunks(self, text, metadata):
#         """
#         Create chunks from text using a hybrid approach:
#         1. Split the document by detected legal sections.
#         2. Use RecursiveCharacterTextSplitter on the content of each section.
#         """
#         sections = self.detect_legal_sections(text)
#         all_chunks = []
#         last_pos = 0

#         # Handle text before the first section
#         if sections:
#             first_section_start = sections[0]['position']
#             if first_section_start > 0:
#                 initial_text = text[:first_section_start].strip()
#                 if initial_text:
#                     # Split the initial text and add chunks
#                     split_texts = self.text_splitter.split_text(initial_text)
#                     for i, chunk_text in enumerate(split_texts):
#                         all_chunks.append({
#                             'text': chunk_text,
#                             'metadata': {**metadata, 'chunk_index': len(all_chunks)}
#                         })
#             last_pos = first_section_start

#         # Process each section
#         for i, section in enumerate(sections):
#             start = section['position']
#             end = sections[i + 1]['position'] if i + 1 < len(sections) else len(text)
#             section_text = text[start:end].strip()

#             section_metadata = {
#                 **metadata,
#                 'section_type': section['type'],
#                 'section_number': section['number']
#             }

#             # Use the robust splitter on the text within this section
#             split_texts = self.text_splitter.split_text(section_text)
#             for chunk_text in split_texts:
#                 all_chunks.append({
#                     'text': chunk_text,
#                     'metadata': {**section_metadata, 'chunk_index': len(all_chunks)}
#                 })

#         # If no sections were found at all, split the entire document
#         if not sections:
#             split_texts = self.text_splitter.split_text(text)
#             for i, chunk_text in enumerate(split_texts):
#                 all_chunks.append({
#                     'text': chunk_text,
#                     'metadata': {**metadata, 'chunk_index': i}
#                 })

#         return all_chunks

#     def process_volume(self, volume_folder):
#         """Process all PDFs in a volume folder"""
#         volume_folder = Path(volume_folder)
#         print(f"\n📂 Processing: {volume_folder.name}")
#         pdf_files = list(volume_folder.glob("*.pdf"))
#         print(f"   Found {len(pdf_files)} PDF files")

#         all_chunks = []
#         failed_files = []

#         for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
#             text, num_pages = self.extract_text_from_pdf(pdf_file)
#             if not text:
#                 failed_files.append(pdf_file.name)
#                 continue

#             metadata = {
#                 'filename': pdf_file.name,
#                 'volume': volume_folder.name,
#                 'total_pages': num_pages,
#                 'source': 'law_commission'
#             }
#             chunks = self.create_chunks(text, metadata)
#             all_chunks.extend(chunks)

#         print(f"   ✓ Created {len(all_chunks)} chunks")
#         if failed_files:
#             print(f"   ⚠ Failed: {len(failed_files)} files")

#         return all_chunks, failed_files

# def main():
#     print("=" * 70)
#     print("   STEP 1: PROCESSING PDFs INTO CHUNKS (IMPROVED VERSION)")
#     print("=" * 70)

#     # Initialize processor with token-based chunking
#     # 1000 tokens is a safe and effective size. Overlap helps maintain context.
#     processor = SimplePDFProcessor(chunk_size=1000, chunk_overlap=150)

#     data_folder = Path("data")
#     volume_folders = sorted(data_folder.glob("volume_*"))
#     if not volume_folders:
#         exclude_folders = {'processed', 'embeddings', '__pycache__'}
#         volume_folders = sorted([d for d in data_folder.iterdir() if d.is_dir() and d.name not in exclude_folders])

#     if not volume_folders:
#         print("❌ No volume folders found in 'data/' directory. Please check the folder structure.")
#         return

#     print(f"\nFound {len(volume_folders)} volume(s):")
#     for vol in volume_folders:
#         print(f"  - {vol.name}")

#     all_chunks = []
#     all_failed = []
#     for volume_folder in volume_folders:
#         chunks, failed = processor.process_volume(volume_folder)
#         all_chunks.extend(chunks)
#         all_failed.extend(failed)

#     output_dir = Path("data/processed")
#     output_dir.mkdir(exist_ok=True)
#     output_file = output_dir / "all_chunks.json"
#     with open(output_file, 'w', encoding='utf-8') as f:
#         json.dump(all_chunks, f, indent=2, ensure_ascii=False)

#     avg_chunk_size_tokens = 0
#     if all_chunks:
#         enc = tiktoken.get_encoding("cl100k_base")
#         total_tokens = sum(len(enc.encode(c['text'])) for c in all_chunks)
#         avg_chunk_size_tokens = total_tokens / len(all_chunks)

#     stats = {
#         'total_chunks': len(all_chunks),
#         'total_failed_files': len(all_failed),
#         'failed_files': all_failed,
#         'volumes_processed': [v.name for v in volume_folders],
#         'avg_chunk_size_tokens': avg_chunk_size_tokens
#     }
#     stats_file = output_dir / "processing_stats.json"
#     with open(stats_file, 'w', encoding='utf-8') as f:
#         json.dump(stats, f, indent=2, ensure_ascii=False)

#     print("\n" + "=" * 70)
#     print("   PROCESSING COMPLETE!")
#     print("=" * 70)
#     print(f"✓ Total chunks created: {len(all_chunks)}")
#     print(f"✓ Average chunk size: {stats['avg_chunk_size_tokens']:.0f} tokens")
#     print(f"✓ Output saved to: {output_file}")
#     print(f"✓ Stats saved to: {stats_file}")
#     if all_failed:
#         print(f"\n⚠ Warning: {len(all_failed)} files failed to process")
#     print("\n🎯 Next step: Run step2_generate_embeddings.py")


# if __name__ == "__main__":
#     main()




"""
STEP 1: Process PDFs with Citation Metadata
"""
import fitz
import json
import re
from pathlib import Path
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

class SimplePDFProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=150):
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="text-embedding-3-large",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF with page markers"""
        doc = None
        try:
            doc = fitz.open(pdf_path)
            full_text = ""
            for page_num, page in enumerate(doc):
                full_text += page.get_text() + f"\n[PAGE {page_num + 1}]\n"
            return full_text, len(doc)
        except Exception as e:
            print(f"❌ Error processing {pdf_path.name}: {e}")
            return None, 0
        finally:
            if doc:
                doc.close()

    def detect_legal_sections(self, text):
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
                sections.append({
                    'type': section_type,
                    'number': match.group(1),
                    'position': match.start()
                })
        return sorted(sections, key=lambda x: x['position'])

    def extract_page_numbers(self, text):
        """Extract page numbers from [PAGE X] markers"""
        page_markers = re.findall(r'\[PAGE (\d+)\]', text)
        if page_markers:
            return sorted(list(set(int(p) for p in page_markers)))
        return []

    def format_citation(self, metadata, section=None, page_numbers=None):
        """Format citation string"""
        parts = []
        doc_name = metadata.get('filename', '').replace('.pdf', '')
        doc_name = doc_name.replace('-', ' ').replace('_', ' ')
        parts.append(doc_name)
        
        if section:
            section_str = f"{section['type'].replace('_nepali', '').title()} {section['number']}"
            parts.append(section_str)
        
        if page_numbers and len(page_numbers) > 0:
            if len(page_numbers) == 1:
                parts.append(f"p. {page_numbers[0]}")
            else:
                parts.append(f"pp. {min(page_numbers)}-{max(page_numbers)}")
        
        return ", ".join(parts)

    def create_chunks(self, text, metadata):
        """Create chunks with citation metadata"""
        sections = self.detect_legal_sections(text)
        all_chunks = []

        if sections:
            first_section_start = sections[0]['position']
            if first_section_start > 0:
                initial_text = text[:first_section_start].strip()
                if initial_text:
                    page_numbers = self.extract_page_numbers(initial_text)
                    split_texts = self.text_splitter.split_text(initial_text)
                    
                    for i, chunk_text in enumerate(split_texts):
                        chunk_pages = self.extract_page_numbers(chunk_text)
                        all_chunks.append({
                            'text': chunk_text,
                            'metadata': {
                                **metadata,
                                'chunk_index': len(all_chunks),
                                'page_numbers': chunk_pages,
                                'citation': self.format_citation(metadata, None, chunk_pages)
                            }
                        })

        for i, section in enumerate(sections):
            start = section['position']
            end = sections[i + 1]['position'] if i + 1 < len(sections) else len(text)
            section_text = text[start:end].strip()
            section_pages = self.extract_page_numbers(section_text)

            section_metadata = {
                **metadata,
                'section_type': section['type'],
                'section_number': section['number'],
                'page_numbers': section_pages,
            }

            split_texts = self.text_splitter.split_text(section_text)
            
            for sub_idx, chunk_text in enumerate(split_texts):
                chunk_pages = self.extract_page_numbers(chunk_text)
                if not chunk_pages and section_pages:
                    chunk_pages = section_pages
                
                all_chunks.append({
                    'text': chunk_text,
                    'metadata': {
                        **section_metadata,
                        'chunk_index': len(all_chunks),
                        'sub_section_index': sub_idx,
                        'page_numbers': chunk_pages,
                        'citation': self.format_citation(metadata, section, chunk_pages)
                    }
                })

        if not sections:
            split_texts = self.text_splitter.split_text(text)
            for i, chunk_text in enumerate(split_texts):
                chunk_pages = self.extract_page_numbers(chunk_text)
                all_chunks.append({
                    'text': chunk_text,
                    'metadata': {
                        **metadata,
                        'chunk_index': i,
                        'page_numbers': chunk_pages,
                        'citation': self.format_citation(metadata, None, chunk_pages)
                    }
                })

        return all_chunks

    def process_volume(self, volume_folder):
        volume_folder = Path(volume_folder)
        print(f"\n📂 Processing: {volume_folder.name}")
        pdf_files = list(volume_folder.glob("*.pdf"))
        print(f"   Found {len(pdf_files)} PDF files")

        all_chunks = []
        failed_files = []

        for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
            text, num_pages = self.extract_text_from_pdf(pdf_file)
            if not text:
                failed_files.append(pdf_file.name)
                continue

            metadata = {
                'filename': pdf_file.name,
                'volume': volume_folder.name,
                'total_pages': num_pages,
                'source': 'law_commission'
            }
            chunks = self.create_chunks(text, metadata)
            all_chunks.extend(chunks)

        print(f"   ✓ Created {len(all_chunks)} chunks")
        if failed_files:
            print(f"   ⚠ Failed: {len(failed_files)} files")

        return all_chunks, failed_files

def main():
    print("=" * 70)
    print("   STEP 1: PROCESSING PDFs WITH CITATIONS")
    print("=" * 70)

    processor = SimplePDFProcessor(chunk_size=1000, chunk_overlap=150)

    data_folder = Path("data")
    volume_folders = sorted(data_folder.glob("volume_*"))
    if not volume_folders:
        exclude_folders = {'processed', 'embeddings', '__pycache__'}
        volume_folders = sorted([d for d in data_folder.iterdir() if d.is_dir() and d.name not in exclude_folders])

    if not volume_folders:
        print("❌ No volume folders found")
        return

    print(f"\nFound {len(volume_folders)} volume(s)")

    all_chunks = []
    all_failed = []
    for volume_folder in volume_folders:
        chunks, failed = processor.process_volume(volume_folder)
        all_chunks.extend(chunks)
        all_failed.extend(failed)

    output_dir = Path("data/processed")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "all_chunks.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    # Calculate stats
    chunks_with_sections = sum(1 for c in all_chunks if c['metadata'].get('section_type'))
    chunks_with_pages = sum(1 for c in all_chunks if c['metadata'].get('page_numbers'))
    
    if all_chunks:
        enc = tiktoken.get_encoding("cl100k_base")
        total_tokens = sum(len(enc.encode(c['text'])) for c in all_chunks)
        avg_chunk_size_tokens = total_tokens / len(all_chunks)
    else:
        avg_chunk_size_tokens = 0

    stats = {
        'total_chunks': len(all_chunks),
        'chunks_with_sections': chunks_with_sections,
        'chunks_with_page_numbers': chunks_with_pages,
        'total_failed_files': len(all_failed),
        'failed_files': all_failed,
        'volumes_processed': [v.name for v in volume_folders],
        'avg_chunk_size_tokens': avg_chunk_size_tokens
    }
    
    stats_file = output_dir / "processing_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("   PROCESSING COMPLETE!")
    print("=" * 70)
    print(f"✓ Total chunks: {len(all_chunks)}")
    print(f"✓ With sections: {chunks_with_sections} ({chunks_with_sections/len(all_chunks)*100:.1f}%)")
    print(f"✓ With pages: {chunks_with_pages} ({chunks_with_pages/len(all_chunks)*100:.1f}%)")
    print(f"✓ Avg size: {avg_chunk_size_tokens:.0f} tokens")
    
    if all_chunks and all_chunks[0]['metadata'].get('citation'):
        print(f"\n📖 Sample citation: {all_chunks[0]['metadata']['citation']}")
    
    print("\n🎯 Next: Run step2_generate_embeddings.py")

if __name__ == "__main__":
    main()