import re
import json
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import fitz  # PyMuPDF
from pathlib import Path
import os

@dataclass
class DocumentStructure:
    """Represents the hierarchical structure of a document"""
    section_number: str
    title: str
    level: int
    parent_section: Optional[str]
    page_number: int
    start_position: int
    content: str = ""
    children: List[str] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []

@dataclass
class ChunkMetadata:
    """Simple metadata for each chunk"""
    chunk_id: str
    document_name: str
    section_number: str
    section_title: str
    page_number: int
    chunk_type: str  # 'small' or 'large'
    chunk_size: int

@dataclass
class ProcessedChunk:
    """A chunk with its content and metadata"""
    content: str
    metadata: ChunkMetadata
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'content': self.content,
            'metadata': asdict(self.metadata)
        }

class PDFStructureAnalyzer:
    """Simple PDF structure analyzer"""
    
    def __init__(self):
        # Simple patterns for section numbering
        self.section_patterns = [
            r'^(\d+(?:\.\d+)*)\s+(.+?)(?:\s*\.{2,}|\s*$)',  # 1, 1.1, 1.1.1
            r'^([A-Z]\.\d+(?:\.\d+)*)\s+(.+?)(?:\s*\.{2,}|\s*$)',  # A.1, A.1.1
        ]

    def extract_pdf_content(self, pdf_path: str) -> List[Dict]:
        """Extract text content from PDF with page information"""
        doc = fitz.open(pdf_path)
        pages_content = []
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text = page.get_text()
            
            pages_content.append({
                'page_number': page_num + 1,
                'text': text
            })
        
        doc.close()
        return pages_content

    def identify_section_structure(self, pages_content: List[Dict]) -> Dict[str, DocumentStructure]:
        """Identify the hierarchical structure of the document"""
        sections = {}
        
        for page_data in pages_content:
            page_num = page_data['page_number']
            text = page_data['text']
            lines = text.split('\n')
            
            for line_idx, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                # Try to match section patterns
                for pattern in self.section_patterns:
                    match = re.match(pattern, line, re.IGNORECASE)
                    if match:
                        section_num = match.group(1)
                        title = match.group(2).strip()
                        
                        # Calculate level based on dots
                        level = section_num.count('.') + 1
                        
                        # Find parent section
                        parent_section = None
                        if level > 1:
                            parent_parts = section_num.split('.')
                            parent_section = '.'.join(parent_parts[:-1])
                        
                        # Create structure
                        structure = DocumentStructure(
                            section_number=section_num,
                            title=title,
                            level=level,
                            parent_section=parent_section,
                            page_number=page_num,
                            start_position=line_idx
                        )
                        
                        sections[section_num] = structure
                        
                        # Update parent-child relationships
                        if parent_section and parent_section in sections:
                            sections[parent_section].children.append(section_num)
                        
                        break
        
        return sections

    def extract_section_content(self, pages_content: List[Dict], sections: Dict[str, DocumentStructure]) -> Dict[str, DocumentStructure]:
        """Extract content for each section"""
        section_list = sorted(sections.values(), key=lambda x: (x.page_number, x.start_position))
        
        for i, section in enumerate(section_list):
            content_lines = []
            start_page = section.page_number - 1  # Convert to 0-based
            start_line = section.start_position
            
            # Determine end point
            if i + 1 < len(section_list):
                next_section = section_list[i + 1]
                end_page = next_section.page_number - 1
                end_line = next_section.start_position
            else:
                end_page = len(pages_content) - 1
                end_line = float('inf')
            
            # Extract content
            for page_idx in range(start_page, min(end_page + 1, len(pages_content))):
                page_text = pages_content[page_idx]['text']
                lines = page_text.split('\n')
                
                start_idx = start_line if page_idx == start_page else 0
                end_idx = end_line if page_idx == end_page else len(lines)
                
                for line_idx in range(int(start_idx), min(int(end_idx), len(lines))):
                    if line_idx < len(lines):
                        content_lines.append(lines[line_idx])
            
            section.content = '\n'.join(content_lines)
            sections[section.section_number] = section
        
        return sections

class SimpleChunker:
    """Simple chunker that creates small and large chunks"""
    
    def __init__(self, small_chunk_size: int = 512, large_chunk_size: int = 2048):
        self.small_chunk_size = small_chunk_size
        self.large_chunk_size = large_chunk_size
        self.analyzer = PDFStructureAnalyzer()
    
    def create_chunks(self, pdf_path: str) -> List[ProcessedChunk]:
        """Main method to create chunks from PDF"""
        document_name = Path(pdf_path).stem
        
        # Extract PDF content
        print("Extracting PDF content...")
        pages_content = self.analyzer.extract_pdf_content(pdf_path)
        
        # Identify structure
        print("Analyzing document structure...")
        sections = self.analyzer.identify_section_structure(pages_content)
        
        # Extract section content
        print("Extracting section content...")
        sections = self.analyzer.extract_section_content(pages_content, sections)
        
        # Create chunks
        print("Creating chunks...")
        all_chunks = []
        
        for section_num, section in sections.items():
            if not section.content.strip():
                continue
            
            # Create small chunks
            small_chunks = self._create_chunks(section, document_name, 'small', self.small_chunk_size)
            all_chunks.extend(small_chunks)
            
            # Create large chunks
            large_chunks = self._create_chunks(section, document_name, 'large', self.large_chunk_size)
            all_chunks.extend(large_chunks)
        
        print(f"Created {len(all_chunks)} chunks total")
        return all_chunks
    
    def _create_chunks(self, section: DocumentStructure, document_name: str, chunk_type: str, max_size: int) -> List[ProcessedChunk]:
        """Create chunks of specified type and size"""
        chunks = []
        content = section.content
        
        # Split content into chunks
        text_chunks = self._split_text_smart(content, max_size)
        
        for i, chunk_text in enumerate(text_chunks):
            chunk_id = f"{section.section_number}_{chunk_type}_{i+1:03d}"
            
            metadata = ChunkMetadata(
                chunk_id=chunk_id,
                document_name=document_name,
                section_number=section.section_number,
                section_title=section.title,
                page_number=section.page_number,
                chunk_type=chunk_type,
                chunk_size=len(chunk_text)
            )
            
            chunks.append(ProcessedChunk(chunk_text, metadata))
        
        return chunks
    
    def _split_text_smart(self, text: str, max_size: int) -> List[str]:
        """Smart text splitting that respects sentence and paragraph boundaries"""
        if len(text) <= max_size:
            return [text]
        
        chunks = []
        current_chunk = ""
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) + 2 <= max_size:
                if current_chunk:
                    current_chunk += '\n\n' + paragraph
                else:
                    current_chunk = paragraph
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                
                # If paragraph itself is too long, split by sentences
                if len(paragraph) > max_size:
                    sentences = re.split(r'(?<=\.)\s+', paragraph)
                    current_chunk = ""
                    
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) + 1 <= max_size:
                            if current_chunk:
                                current_chunk += ' ' + sentence
                            else:
                                current_chunk = sentence
                        else:
                            if current_chunk:
                                chunks.append(current_chunk)
                            current_chunk = sentence
                else:
                    current_chunk = paragraph
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def save_chunks_as_text_files(self, chunks: List[ProcessedChunk], output_dir: str):
        """Save chunks as two separate text files - one for small, one for large"""
        os.makedirs(output_dir, exist_ok=True)
        
        small_chunks_file = os.path.join(output_dir, "small_chunks.txt")
        large_chunks_file = os.path.join(output_dir, "large_chunks.txt")
        
        # Separate chunks by type
        small_chunks = [c for c in chunks if c.metadata.chunk_type == 'small']
        large_chunks = [c for c in chunks if c.metadata.chunk_type == 'large']
        
        # Sort chunks by section number for better organization
        small_chunks.sort(key=lambda x: self._sort_key(x.metadata.section_number))
        large_chunks.sort(key=lambda x: self._sort_key(x.metadata.section_number))
        
        # Write small chunks file
        with open(small_chunks_file, 'w', encoding='utf-8') as f:
            f.write(f"SMALL CHUNKS ({self.small_chunk_size} chars max)\n")
            f.write("="*80 + "\n\n")
            
            for chunk in small_chunks:
                f.write(f"CHUNK: {chunk.metadata.chunk_id}\n")
                f.write(f"SECTION: {chunk.metadata.section_number} - {chunk.metadata.section_title}\n")
                f.write(f"PAGE: {chunk.metadata.page_number}\n")
                f.write(f"SIZE: {chunk.metadata.chunk_size} chars\n")
                f.write("-" * 50 + "\n")
                f.write(chunk.content)
                f.write("\n" + "="*80 + "\n\n")
        
        # Write large chunks file
        with open(large_chunks_file, 'w', encoding='utf-8') as f:
            f.write(f"LARGE CHUNKS ({self.large_chunk_size} chars max)\n")
            f.write("="*80 + "\n\n")
            
            for chunk in large_chunks:
                f.write(f"CHUNK: {chunk.metadata.chunk_id}\n")
                f.write(f"SECTION: {chunk.metadata.section_number} - {chunk.metadata.section_title}\n")
                f.write(f"PAGE: {chunk.metadata.page_number}\n")
                f.write(f"SIZE: {chunk.metadata.chunk_size} chars\n")
                f.write("-" * 50 + "\n")
                f.write(chunk.content)
                f.write("\n" + "="*80 + "\n\n")
        
        print(f"\n✅ CHUNKS SAVED:")
        print(f"📄 Small chunks: {small_chunks_file} ({len(small_chunks)} chunks)")
        print(f"📄 Large chunks: {large_chunks_file} ({len(large_chunks)} chunks)")
    
    def _sort_key(self, section_number: str) -> tuple:
        """Create sort key for section numbers like 7.1.2"""
        parts = section_number.split('.')
        return tuple(int(part) if part.isdigit() else part for part in parts)
    
    def print_summary(self, chunks: List[ProcessedChunk]):
        """Print simple summary"""
        small_chunks = [c for c in chunks if c.metadata.chunk_type == 'small']
        large_chunks = [c for c in chunks if c.metadata.chunk_type == 'large']
        
        sections = defaultdict(list)
        for chunk in chunks:
            sections[chunk.metadata.section_number].append(chunk)
        
        print(f"\n{'='*60}")
        print("PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Total chunks: {len(chunks)}")
        print(f"Small chunks: {len(small_chunks)}")
        print(f"Large chunks: {len(large_chunks)}")
        print(f"Sections found: {len(sections)}")
        print(f"{'='*60}")
        
        print("\nSECTIONS:")
        for section_num in sorted(sections.keys(), key=self._sort_key):
            section_chunks = sections[section_num]
            sample_chunk = section_chunks[0]
            small_count = sum(1 for c in section_chunks if c.metadata.chunk_type == 'small')
            large_count = sum(1 for c in section_chunks if c.metadata.chunk_type == 'large')
            
            print(f"  {section_num} - {sample_chunk.metadata.section_title}")
            print(f"    Page: {sample_chunk.metadata.page_number}")
            print(f"    Chunks: {small_count} small, {large_count} large")

# Example usage
if __name__ == "__main__":
    # Initialize the chunker
    chunker = SimpleChunker(small_chunk_size=512, large_chunk_size=2048)
    
    # Process the PDF
    pdf_path = "path/to/your/manual.pdf"  # Replace with actual path
    
    try:
        # Create chunks
        chunks = chunker.create_chunks(pdf_path)
        
        # Print summary
        chunker.print_summary(chunks)
        
        # Save chunks as text files
        output_dir = "processed_chunks"
        chunker.save_chunks_as_text_files(chunks, output_dir)
        
        print(f"\n{'='*60}")
        print("✅ PROCESSING COMPLETE!")
        print(f"📁 Output directory: {output_dir}")
        print(f"📄 Small chunks file: small_chunks.txt")
        print(f"📄 Large chunks file: large_chunks.txt")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
        print("Please ensure PyMuPDF is installed: pip install PyMuPDF")
        import traceback
        traceback.print_exc()
