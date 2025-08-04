import re
import json
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import fitz  # PyMuPDF
from pathlib import Path

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
    """Rich metadata for each chunk"""
    chunk_id: str
    document_name: str
    section_hierarchy: List[str]
    section_titles: List[str]
    section_number: str
    section_title: str
    page_number: int
    chunk_type: str  # 'micro' or 'macro'
    chunk_size: int
    step_numbers: List[int]
    subsection: str
    procedure_type: str
    equipment_mentioned: List[str]
    safety_warnings: List[str]
    measurements: List[str]
    cross_references: List[str]
    has_diagrams: bool
    keywords: List[str]

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
    """Intelligently analyzes PDF structure and extracts hierarchical information"""
    
    def __init__(self):
        # Patterns for different section numbering schemes
        self.section_patterns = [
            # Standard numeric: 1, 1.1, 1.1.1, 1.1.1.1
            r'^(\d+(?:\.\d+)*)\s+(.+?)(?:\s*\.{2,}|\s*$)',
            # With appendix: A.1, A.1.1
            r'^([A-Z]\.\d+(?:\.\d+)*)\s+(.+?)(?:\s*\.{2,}|\s*$)',
            # Roman numerals: I, II, III
            r'^([IVX]+)\.\s+(.+?)(?:\s*\.{2,}|\s*$)',
            # Mixed: 7.1.2.1, 7.1.2.2
            r'^(\d+(?:\.\d+)+)\s+(.+?)(?:\s*\.{2,}|\s*$)',
        ]
        
        # Step patterns
        self.step_patterns = [
            r'Step\s+(\d+)',
            r'^\s*(\d+)\.\s+',
            r'^\s*\((\d+)\)\s+',
        ]
        
        # Safety warning patterns
        self.safety_patterns = [
            r'WARNING\s*[:\-]?\s*(.+?)(?=\n\n|\n[A-Z]|$)',
            r'CAUTION\s*[:\-]?\s*(.+?)(?=\n\n|\n[A-Z]|$)',
            r'NOTICE\s*[:\-]?\s*(.+?)(?=\n\n|\n[A-Z]|$)',
            r'DANGER\s*[:\-]?\s*(.+?)(?=\n\n|\n[A-Z]|$)',
        ]
        
        # Equipment/tools patterns
        self.equipment_patterns = [
            r'(?:tool|equipment|wrench|jack|pump|filter|motor|bearing|disc|shaft|bolt|nut)s?',
            r'\d+\s*(?:mm|cm|m|kg|t|Nm|°C|μm|micron)',
            r'M\d+(?:\s*×\s*\d+)?(?:\s*mm)?',  # Bolt specifications
        ]
        
        # Cross-reference patterns
        self.cross_ref_patterns = [
            r'refer\s+to\s+["\']?([^"\']+)["\']?',
            r'see\s+(?:chapter|section)\s+([^\s]+)',
            r'\[}\s*(\d+)\s*\]',  # [} 99] style references
        ]

    def extract_pdf_content(self, pdf_path: str) -> List[Dict]:
        """Extract text content from PDF with page information"""
        doc = fitz.open(pdf_path)
        pages_content = []
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text = page.get_text()
            
            # Get text blocks with position information
            blocks = page.get_text("dict")
            
            pages_content.append({
                'page_number': page_num + 1,
                'text': text,
                'blocks': blocks
            })
        
        doc.close()
        return pages_content

    def identify_section_structure(self, pages_content: List[Dict]) -> Dict[str, DocumentStructure]:
        """Identify the hierarchical structure of the document"""
        sections = {}
        current_hierarchy = {}
        
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
                        current_hierarchy[level] = section_num
                        
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

    def extract_rich_metadata(self, text: str, section: DocumentStructure, document_name: str) -> Dict[str, Any]:
        """Extract rich metadata from text content"""
        metadata = {
            'step_numbers': [],
            'equipment_mentioned': [],
            'safety_warnings': [],
            'measurements': [],
            'cross_references': [],
            'keywords': [],
            'has_diagrams': False
        }
        
        # Extract step numbers
        for pattern in self.step_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            metadata['step_numbers'].extend([int(m) for m in matches if m.isdigit()])
        
        # Extract safety warnings
        for pattern in self.safety_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            metadata['safety_warnings'].extend(matches)
        
        # Extract equipment mentions
        equipment_matches = re.findall('|'.join(self.equipment_patterns), text, re.IGNORECASE)
        metadata['equipment_mentioned'] = list(set(equipment_matches))
        
        # Extract measurements and specifications
        measurement_patterns = [
            r'\d+(?:\.\d+)?\s*(?:mm|cm|m|kg|t|Nm|°C|μm|micron)',
            r'M\d+(?:\s*×\s*\d+)?',
            r'\d+:\d+',  # Ratios
        ]
        for pattern in measurement_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            metadata['measurements'].extend(matches)
        
        # Extract cross-references
        for pattern in self.cross_ref_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            metadata['cross_references'].extend(matches)
        
        # Check for diagrams/figures
        diagram_indicators = ['figure', 'diagram', 'image', 'illustration', 'drawing']
        metadata['has_diagrams'] = any(indicator in text.lower() for indicator in diagram_indicators)
        
        # Extract keywords (simple approach - can be enhanced with NLP)
        # Remove common words and extract meaningful terms
        import re
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        common_words = {'with', 'from', 'that', 'this', 'have', 'will', 'been', 'were', 'said', 
                       'each', 'which', 'their', 'time', 'into', 'only', 'other', 'after', 'first', 
                       'well', 'also', 'where', 'much', 'before', 'here', 'through', 'when', 'should'}
        meaningful_words = [w for w in set(words) if w not in common_words and len(w) > 3]
        metadata['keywords'] = meaningful_words[:20]  # Top 20 keywords
        
        return metadata

class HierarchicalChunker:
    """Creates micro and macro chunks with rich metadata"""
    
    def __init__(self, micro_chunk_size: int = 512, macro_chunk_size: int = 2048):
        self.micro_chunk_size = micro_chunk_size
        self.macro_chunk_size = macro_chunk_size
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
            
            # Create micro chunks
            micro_chunks = self._create_micro_chunks(section, document_name)
            all_chunks.extend(micro_chunks)
            
            # Create macro chunks
            macro_chunks = self._create_macro_chunks(section, document_name)
            all_chunks.extend(macro_chunks)
        
        print(f"Created {len(all_chunks)} chunks total")
        return all_chunks
    
    def _create_micro_chunks(self, section: DocumentStructure, document_name: str) -> List[ProcessedChunk]:
        """Create small, focused chunks"""
        chunks = []
        content = section.content
        
        # Try to split by steps first
        step_splits = re.split(r'\n(?=Step\s+\d+)', content)
        
        if len(step_splits) > 1:
            # Process step-based chunks
            for i, step_content in enumerate(step_splits):
                if not step_content.strip():
                    continue
                
                # Further split if step is too long
                step_chunks = self._split_text_smart(step_content, self.micro_chunk_size)
                
                for j, chunk_text in enumerate(step_chunks):
                    chunk_id = f"{section.section_number}_micro_step_{i}_{j}"
                    metadata = self._create_chunk_metadata(
                        chunk_id, document_name, section, chunk_text, 'micro'
                    )
                    chunks.append(ProcessedChunk(chunk_text, metadata))
        else:
            # Split by size for non-step content
            text_chunks = self._split_text_smart(content, self.micro_chunk_size)
            
            for i, chunk_text in enumerate(text_chunks):
                chunk_id = f"{section.section_number}_micro_{i}"
                metadata = self._create_chunk_metadata(
                    chunk_id, document_name, section, chunk_text, 'micro'
                )
                chunks.append(ProcessedChunk(chunk_text, metadata))
        
        return chunks
    
    def _create_macro_chunks(self, section: DocumentStructure, document_name: str) -> List[ProcessedChunk]:
        """Create larger, contextual chunks"""
        chunks = []
        content = section.content
        
        # Split into larger chunks
        text_chunks = self._split_text_smart(content, self.macro_chunk_size)
        
        for i, chunk_text in enumerate(text_chunks):
            chunk_id = f"{section.section_number}_macro_{i}"
            metadata = self._create_chunk_metadata(
                chunk_id, document_name, section, chunk_text, 'macro'
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
    
    def _create_chunk_metadata(self, chunk_id: str, document_name: str, 
                             section: DocumentStructure, content: str, chunk_type: str) -> ChunkMetadata:
        """Create rich metadata for a chunk"""
        
        # Build section hierarchy
        hierarchy = [section.section_number]
        titles = [section.title]
        
        # Add parent sections
        current_section = section.parent_section
        while current_section:
            hierarchy.insert(0, current_section)
            # You might want to store parent titles separately
            current_section = None  # Simplified for now
        
        # Extract metadata from content
        rich_metadata = self.analyzer.extract_rich_metadata(content, section, document_name)
        
        # Determine procedure type based on section title
        procedure_type = self._determine_procedure_type(section.title)
        
        return ChunkMetadata(
            chunk_id=chunk_id,
            document_name=document_name,
            section_hierarchy=hierarchy,
            section_titles=titles,
            section_number=section.section_number,
            section_title=section.title,
            page_number=section.page_number,
            chunk_type=chunk_type,
            chunk_size=len(content),
            step_numbers=rich_metadata['step_numbers'],
            subsection=section.section_number,
            procedure_type=procedure_type,
            equipment_mentioned=rich_metadata['equipment_mentioned'],
            safety_warnings=rich_metadata['safety_warnings'],
            measurements=rich_metadata['measurements'],
            cross_references=rich_metadata['cross_references'],
            has_diagrams=rich_metadata['has_diagrams'],
            keywords=rich_metadata['keywords']
        )
    
    def _determine_procedure_type(self, title: str) -> str:
        """Determine the type of procedure based on title"""
        title_lower = title.lower()
        
        if 'oil' in title_lower:
            if 'replacement' in title_lower or 'exchange' in title_lower:
                return 'oil_replacement'
            elif 'drain' in title_lower:
                return 'oil_draining'
            elif 'fill' in title_lower:
                return 'oil_filling'
            elif 'sample' in title_lower:
                return 'oil_sampling'
            else:
                return 'oil_maintenance'
        elif 'filter' in title_lower:
            return 'filter_maintenance'
        elif 'gearbox' in title_lower:
            if 'replacement' in title_lower:
                return 'gearbox_replacement'
            else:
                return 'gearbox_maintenance'
        elif 'cooler' in title_lower:
            return 'cooler_maintenance'
        elif 'pump' in title_lower:
            return 'pump_maintenance'
        else:
            return 'general_procedure'

    def save_chunks_to_json(self, chunks: List[ProcessedChunk], output_path: str):
        """Save processed chunks to JSON file"""
        chunks_data = [chunk.to_dict() for chunk in chunks]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(chunks)} chunks to {output_path}")
    
    def save_chunks_as_text_files(self, chunks: List[ProcessedChunk], output_dir: str):
        """Save each chunk as a separate text file for easy inspection"""
        import os
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subdirectories for micro and macro chunks
        micro_dir = os.path.join(output_dir, "micro_chunks")
        macro_dir = os.path.join(output_dir, "macro_chunks")
        os.makedirs(micro_dir, exist_ok=True)
        os.makedirs(macro_dir, exist_ok=True)
        
        # Group chunks by section for better organization
        section_groups = defaultdict(list)
        for chunk in chunks:
            section_key = chunk.metadata.section_
