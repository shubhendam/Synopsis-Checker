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
            section_key = chunk.metadata.section_number
            section_groups[section_key].append(chunk)
        
        chunk_count = 0
        
        for section_num, section_chunks in section_groups.items():
            # Sort chunks by chunk_id to maintain order
            section_chunks.sort(key=lambda x: x.metadata.chunk_id)
            
            for chunk in section_chunks:
                chunk_count += 1
                
                # Determine output directory based on chunk type
                target_dir = micro_dir if chunk.metadata.chunk_type == 'micro' else macro_dir
                
                # Create safe filename
                safe_section = section_num.replace('.', '_')
                chunk_type = chunk.metadata.chunk_type
                filename = f"{safe_section}_{chunk_type}_{chunk_count:04d}.txt"
                filepath = os.path.join(target_dir, filename)
                
                # Prepare content with metadata header
                content = self._format_chunk_for_text_file(chunk)
                
                # Write to file
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
        
        print(f"\n=== CHUNKS SAVED AS TEXT FILES ===")
        print(f"Total chunks saved: {chunk_count}")
        print(f"Micro chunks directory: {micro_dir}")
        print(f"Macro chunks directory: {macro_dir}")
        print(f"Root directory: {output_dir}")
        
        # Create a summary file
        self._create_chunks_summary_file(chunks, output_dir)
    
    def _format_chunk_for_text_file(self, chunk: ProcessedChunk) -> str:
        """Format chunk content with metadata header for text file"""
        metadata = chunk.metadata
        
        header = f"""{'='*80}
CHUNK METADATA
{'='*80}
Chunk ID: {metadata.chunk_id}
Document: {metadata.document_name}
Section: {metadata.section_number} - {metadata.section_title}
Page: {metadata.page_number}
Type: {metadata.chunk_type.upper()}
Size: {metadata.chunk_size} characters

Section Hierarchy: {' → '.join(metadata.section_hierarchy)}
Section Titles: {' → '.join(metadata.section_titles)}

Procedure Type: {metadata.procedure_type}
Subsection: {metadata.subsection}

Steps Found: {metadata.step_numbers if metadata.step_numbers else 'None'}
Equipment: {', '.join(metadata.equipment_mentioned[:5]) if metadata.equipment_mentioned else 'None'}
Measurements: {', '.join(metadata.measurements[:3]) if metadata.measurements else 'None'}
Cross References: {', '.join(metadata.cross_references[:3]) if metadata.cross_references else 'None'}
Has Diagrams: {'Yes' if metadata.has_diagrams else 'No'}

Safety Warnings: {len(metadata.safety_warnings)} found
{chr(10).join(f"  - {warning[:100]}..." if len(warning) > 100 else f"  - {warning}" for warning in metadata.safety_warnings[:3])}

Keywords: {', '.join(metadata.keywords[:10]) if metadata.keywords else 'None'}

{'='*80}
CHUNK CONTENT
{'='*80}

{chunk.content}

{'='*80}
END OF CHUNK
{'='*80}
"""
        return header
    
    def _create_chunks_summary_file(self, chunks: List[ProcessedChunk], output_dir: str):
        """Create a summary file with overview of all chunks"""
        summary_path = os.path.join(output_dir, "chunks_summary.txt")
        
        # Organize chunks by section
        sections = defaultdict(list)
        for chunk in chunks:
            section_key = f"{chunk.metadata.section_number} - {chunk.metadata.section_title}"
            sections[section_key].append(chunk)
        
        # Count statistics
        total_chunks = len(chunks)
        micro_chunks = sum(1 for c in chunks if c.metadata.chunk_type == 'micro')
        macro_chunks = sum(1 for c in chunks if c.metadata.chunk_type == 'macro')
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"""{'='*100}
DOCUMENT PROCESSING SUMMARY
{'='*100}

Total Chunks Created: {total_chunks}
- Micro Chunks (≤512 chars): {micro_chunks}
- Macro Chunks (≤2048 chars): {macro_chunks}

Document Structure: {len(sections)} sections found

{'='*100}
SECTIONS BREAKDOWN
{'='*100}

""")
            
            for section_key, section_chunks in sorted(sections.items()):
                section_micro = sum(1 for c in section_chunks if c.metadata.chunk_type == 'micro')
                section_macro = sum(1 for c in section_chunks if c.metadata.chunk_type == 'macro')
                
                # Get sample metadata
                sample_chunk = section_chunks[0]
                
                f.write(f"""
{'-'*80}
{section_key}
{'-'*80}
Page: {sample_chunk.metadata.page_number}
Chunks: {len(section_chunks)} total ({section_micro} micro, {section_macro} macro)
Procedure Type: {sample_chunk.metadata.procedure_type}

""")
                
                # Show first few chunks as examples
                f.write("Chunk Examples:\n")
                for i, chunk in enumerate(section_chunks[:3]):
                    f.write(f"  {chunk.metadata.chunk_id}\n")
                    f.write(f"    Type: {chunk.metadata.chunk_type}\n")
                    f.write(f"    Size: {chunk.metadata.chunk_size} chars\n")
                    f.write(f"    Steps: {chunk.metadata.step_numbers}\n")
                    f.write(f"    Content preview: {chunk.content[:100].replace(chr(10), ' ')}...\n")
                    f.write(f"\n")
                
                if len(section_chunks) > 3:
                    f.write(f"  ... and {len(section_chunks) - 3} more chunks\n\n")
            
            f.write(f"\n{'='*100}\n")
            f.write("FILES ORGANIZATION:\n")
            f.write("- micro_chunks/: Contains all micro chunks (≤512 chars)\n")
            f.write("- macro_chunks/: Contains all macro chunks (≤2048 chars)\n")
            f.write("- chunks_summary.txt: This summary file\n")
            f.write("- Each text file contains: metadata header + chunk content\n")
            f.write(f"{'='*100}\n")
        
        print(f"Summary file created: {summary_path}")
    
    def create_test_query_examples(self, chunks: List[ProcessedChunk], output_dir: str):
        """Create example queries and their matching chunks for testing"""
        examples_path = os.path.join(output_dir, "test_queries_examples.txt")
        
        # Define test queries
        test_queries = [
            {
                "query": "step 3 of section 7.1.2",
                "description": "Looking for specific step in specific section",
                "filter_func": lambda c: c.metadata.section_number == '7.1.2' and 3 in c.metadata.step_numbers
            },
            {
                "query": "oil draining procedure",
                "description": "Looking for oil draining procedures",
                "filter_func": lambda c: 'drain' in c.metadata.procedure_type.lower()
            },
            {
                "query": "safety warnings about hot oil",
                "description": "Looking for safety information",
                "filter_func": lambda c: any('hot oil' in warning.lower() for warning in c.metadata.safety_warnings)
            },
            {
                "query": "gearbox replacement steps",
                "description": "Looking for gearbox replacement procedures",
                "filter_func": lambda c: 'replacement' in c.metadata.procedure_type.lower() and c.metadata.step_numbers
            },
            {
                "query": "hydraulic jack usage",
                "description": "Looking for equipment usage",
                "filter_func": lambda c: any('hydraulic' in equip.lower() and 'jack' in equip.lower() for equip in c.metadata.equipment_mentioned)
            }
        ]
        
        with open(examples_path, 'w', encoding='utf-8') as f:
            f.write(f"""{'='*100}
TEST QUERIES AND MATCHING CHUNKS
{'='*100}

This file shows example queries and which chunks would match them.
Use this to test your retrieval system performance.

""")
            
            for i, test_case in enumerate(test_queries, 1):
                matching_chunks = [c for c in chunks if test_case['filter_func'](c)]
                
                f.write(f"""
{'-'*80}
TEST QUERY {i}
{'-'*80}
Query: "{test_case['query']}"
Description: {test_case['description']}
Matching Chunks: {len(matching_chunks)}

""")
                
                if matching_chunks:
                    f.write("Matched Chunks:\n")
                    for chunk in matching_chunks[:5]:  # Show first 5 matches
                        f.write(f"  - {chunk.metadata.chunk_id}\n")
                        f.write(f"    Section: {chunk.metadata.section_number} - {chunk.metadata.section_title}\n")
                        f.write(f"    Type: {chunk.metadata.chunk_type}\n")
                        f.write(f"    Steps: {chunk.metadata.step_numbers}\n")
                        f.write(f"    Preview: {chunk.content[:150].replace(chr(10), ' ')}...\n\n")
                    
                    if len(matching_chunks) > 5:
                        f.write(f"  ... and {len(matching_chunks) - 5} more matches\n")
                else:
                    f.write("No matching chunks found.\n")
                
                f.write(f"\n")
        
        print(f"Test queries examples created: {examples_path}")

    def print_structure_summary(self, chunks: List[ProcessedChunk]):
        """Print a summary of the document structure and chunks"""
        sections = defaultdict(list)
        
        for chunk in chunks:
            section_key = f"{chunk.metadata.section_number} - {chunk.metadata.section_title}"
            sections[section_key].append(chunk)
        
        print("\n=== DOCUMENT STRUCTURE SUMMARY ===")
        for section_key, section_chunks in sections.items():
            micro_count = sum(1 for c in section_chunks if c.metadata.chunk_type == 'micro')
            macro_count = sum(1 for c in section_chunks if c.metadata.chunk_type == 'macro')
            
            print(f"\n{section_key}")
            print(f"  Page: {section_chunks[0].metadata.page_number}")
            print(f"  Chunks: {micro_count} micro, {macro_count} macro")
            
            # Show some metadata examples
            if section_chunks:
                sample_metadata = section_chunks[0].metadata
                if sample_metadata.step_numbers:
                    print(f"  Steps: {sample_metadata.step_numbers}")
                if sample_metadata.equipment_mentioned:
                    print(f"  Equipment: {sample_metadata.equipment_mentioned[:3]}...")
                if sample_metadata.safety_warnings:
                    print(f"  Safety warnings: {len(sample_metadata.safety_warnings)}")

# Example usage
if __name__ == "__main__":
    # Initialize the chunker
    chunker = HierarchicalChunker(micro_chunk_size=512, macro_chunk_size=2048)
    
    # Process the PDF
    pdf_path = "path/to/your/manual.pdf"  # Replace with actual path
    
    try:
        # Create chunks
        chunks = chunker.create_chunks(pdf_path)
        
        # Print summary
        chunker.print_structure_summary(chunks)
        
        # Save chunks as individual text files for inspection
        output_dir = "processed_chunks_text_files"
        chunker.save_chunks_as_text_files(chunks, output_dir)
        
        # Create test query examples
        chunker.create_test_query_examples(chunks, output_dir)
        
        # Also save to JSON (optional)
        json_output_path = "processed_chunks.json"
        chunker.save_chunks_to_json(chunks, json_output_path)
        
        # Example: Find specific chunk
        print("\n=== EXAMPLE QUERIES ===")
        
        # Find chunks from section 7.1.2
        section_chunks = [c for c in chunks if c.metadata.section_number.startswith('7.1.2')]
        print(f"Found {len(section_chunks)} chunks in section 7.1.2")
        
        # Find chunks with step 3
        step3_chunks = [c for c in chunks if 3 in c.metadata.step_numbers]
        print(f"Found {len(step3_chunks)} chunks containing step 3")
        
        # Find chunks with safety warnings
        safety_chunks = [c for c in chunks if c.metadata.safety_warnings]
        print(f"Found {len(safety_chunks)} chunks with safety warnings")
        
        print(f"\n{'='*60}")
        print("PROCESSING COMPLETE!")
        print(f"{'='*60}")
        print(f"✅ Created {len(chunks)} total chunks")
        print(f"📁 Text files saved in: {output_dir}/")
        print(f"📊 Summary file: {output_dir}/chunks_summary.txt")
        print(f"🧪 Test queries: {output_dir}/test_queries_examples.txt")
        print(f"💾 JSON backup: {json_output_path}")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
        print("Please ensure PyMuPDF is installed: pip install PyMuPDF")
        import traceback
        traceback.print_exc()
