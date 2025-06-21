"""
New unified document processing module for IRIS RAG system.
Clean implementation following the hierarchical decision tree.
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer

from .config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    FIRST_PAGE_SAMPLE_CHARS,
    MAX_CHUNK_SIZE,
    MIN_CHUNK_SIZE,
    MIN_LINE_LENGTH,
    TARGET_CHUNK_SIZE,
)


@dataclass
class DocumentStructure:
    """Represents the detected structure of a document."""

    sections: List[Dict]
    tables: List[Dict]
    paragraphs: List[Dict]
    hierarchy_levels: Dict[int, List[Dict]]


@dataclass
class ChunkCandidate:
    """Represents a potential chunk during processing."""

    text: str
    word_count: int
    start_pos: int
    end_pos: int
    structure_type: str  # 'section', 'table', 'paragraph'
    metadata: Dict


def extract_dod_number(
    filename: str, metadata: Dict = None, first_page_text: str = None
) -> str:
    """Extract DOD document number from filename, metadata, or document content."""
    # Strategy 1: Check filename for patterns like "500085p.pdf" -> "5000.85"
    compact_pattern = r"(\d{4})(\d{2})p?\.pdf$"
    match = re.search(compact_pattern, filename, re.IGNORECASE)
    if match:
        return f"{match.group(1)}.{match.group(2)}"

    # Strategy 2: Check filename for standard patterns "DODI 5000.85"
    standard_pattern = r"DOD[IDM][\s_-]?(\d+\.\d+)"
    match = re.search(standard_pattern, filename, re.IGNORECASE)
    if match:
        return match.group(1)

    # Strategy 3: Check metadata title if provided
    if metadata and "Title" in metadata:
        match = re.search(standard_pattern, metadata["Title"], re.IGNORECASE)
        if match:
            return match.group(1)

    # Strategy 4: Check first page text if provided
    if first_page_text:
        # Look for "NUMBER 5000.56" pattern
        number_pattern = r"NUMBER\s+(\d+\.\d+)"
        match = re.search(number_pattern, first_page_text, re.IGNORECASE)
        if match:
            return match.group(1)

        # Fallback to standard pattern in text
        match = re.search(standard_pattern, first_page_text, re.IGNORECASE)
        if match:
            return match.group(1)

    return None


class ConfigurationValidator:
    """Validates chunking configuration."""

    @staticmethod
    def validate():
        """Validate configuration constants."""
        if not (MIN_CHUNK_SIZE <= TARGET_CHUNK_SIZE <= MAX_CHUNK_SIZE):
            raise ValueError(
                f"Invalid chunking configuration: "
                f"MIN_CHUNK_SIZE ({MIN_CHUNK_SIZE}) <= "
                f"TARGET_CHUNK_SIZE ({TARGET_CHUNK_SIZE}) <= "
                f"MAX_CHUNK_SIZE ({MAX_CHUNK_SIZE}) must be true"
            )


class TextCleaner:
    """Handles text cleaning with structure preservation."""

    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)

    def clean_text_preserve_structure(self, text: str) -> str:
        """Clean text while preserving newlines needed for structure detection."""
        if not text:
            return ""

        # Remove table of contents sections
        text = self._remove_table_of_contents(text)

        # Replace multiple spaces/tabs with single space but preserve newlines
        text = re.sub(r"[ \t]+", " ", text)

        # Normalize excessive newlines but keep paragraph breaks
        text = re.sub(r"\n{4,}", "\n\n\n", text)  # Max 3 newlines

        # Remove page breaks and form feeds
        text = re.sub(r"[\f\r\v]", "", text)

        # Remove standalone page numbers (number on its own line)
        text = re.sub(r"\n\s*\d{1,4}\s*\n", "\n", text)

        # Remove common PDF header/footer patterns
        text = re.sub(r"\n\s*Page \d+ of \d+\s*\n", "\n", text, flags=re.IGNORECASE)

        # Remove very short lines but preserve section headers
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            # Keep lines that are:
            # - Longer than minimum OR
            # - Contain important markers OR
            # - Are section headers
            if (
                len(stripped) > MIN_LINE_LENGTH
                or any(
                    marker in stripped.upper()
                    for marker in [
                        "SECTION",
                        "CHAPTER",
                        "PART",
                        "DODD",
                        "DODI",
                        "APPENDIX",
                        "TABLE",
                    ]
                )
                or re.match(r"^[A-Z0-9]+[\.:]\s*[A-Z]", stripped)
                or stripped.startswith("===")  # Table markers
            ):
                cleaned_lines.append(line)  # Preserve original spacing

        text = "\n".join(cleaned_lines)

        # Fix common OCR/extraction errors but preserve URLs
        text = self._fix_ocr_errors_preserve_urls(text)

        return text.strip()

    def final_cleanup(self, text: str) -> str:
        """Final whitespace cleanup after chunking is complete."""
        if not text:
            return ""

        # Remove excess whitespace
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Clean up spacing around punctuation
        text = re.sub(r"\s+([.,;:!?])", r"\1", text)
        text = re.sub(r"([.!?])\s*([a-z])", r"\1 \2", text)

        return text.strip()

    def _remove_table_of_contents(self, text: str) -> str:
        """Remove table of contents sections."""
        if not text:
            return text

        lines = text.split("\n")
        cleaned_lines = []
        in_toc = False
        toc_end_line = -1

        # Find TOC boundaries
        for line_num, line in enumerate(lines):
            line_stripped = line.strip()
            if "TABLE OF CONTENTS" in line_stripped.upper():
                in_toc = True
                self.logger.info("Found and removing TABLE OF CONTENTS section")
            elif in_toc and (
                line_stripped.startswith("SECTION 1:")
                or line_stripped.startswith("1.")
                or "GENERAL ISSUANCE INFORMATION" in line_stripped
                or line_stripped.startswith("CHAPTER")
                or line_stripped.startswith("PART")
                or line_stripped.startswith("APPENDIX")
            ):
                toc_end_line = line_num
                in_toc = False
                break

        # Include lines outside TOC area
        for line_num, line in enumerate(lines):
            if toc_end_line > 0 and line_num < toc_end_line:
                continue

            # Skip TOC remnants
            line_stripped = line.strip()
            if self._is_likely_toc_line(line_stripped):
                continue

            cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    def _is_likely_toc_line(self, line: str) -> bool:
        """Check if a line is likely a table of contents entry."""
        if not line:
            return False

        toc_indicators = [
            r"\.{3,}",  # Multiple dots
            r"\s+\d+$",  # Ending with page number
            r"\.{2,}\s*\d+",  # Dots followed by page number
            r"^\d+\.\d+\..*\d+$",  # "1.1. Something ... 5"
            r"^[A-Z]\.\s+.*\d+$",  # "A. Something ... 5"
        ]

        return any(re.search(pattern, line) for pattern in toc_indicators)

    def _fix_ocr_errors_preserve_urls(self, text: str) -> str:
        """Fix OCR errors while preserving URLs intact."""
        # Find all URLs first
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+[^\s<>"{}|\\^`\[\].,;!?)]'
        urls = re.findall(url_pattern, text)

        # Replace URLs with placeholders
        url_placeholders = {}
        for i, url in enumerate(urls):
            placeholder = f"__URL_PLACEHOLDER_{i}__"
            url_placeholders[placeholder] = url
            text = text.replace(url, placeholder)

        # Fix OCR errors
        text = re.sub(r"\s+([.,;:!?])", r"\1", text)
        text = re.sub(r"([.!?])\s*([a-z])", r"\1 \2", text)

        # Restore URLs
        for placeholder, url in url_placeholders.items():
            text = text.replace(placeholder, url)

        return text


class StructureDetector:
    """Detects document structure (sections, tables, paragraphs)."""

    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)

        # Section patterns for DOD documents
        self.section_patterns = [
            # Higher-level divisions
            {
                "name": "chapter",
                "pattern": r"^CHAPTER\s+\d+",
                "level": 0,
                "priority": "highest",
            },
            {
                "name": "part",
                "pattern": r"^PART\s+[IVX]+",
                "level": 0,
                "priority": "highest",
            },
            # Major sections
            {
                "name": "major_section",
                "pattern": r"^SECTION\s+\d+:",
                "level": 1,
                "priority": "high",
            },
            {
                "name": "appendix",
                "pattern": r"^APPENDIX\s+[A-Z]",
                "level": 1,
                "priority": "high",
            },
            {
                "name": "enclosure",
                "pattern": r"^ENCLOSURE\s+\d+:",
                "level": 1,
                "priority": "high",
            },
            # Numbered sections
            {
                "name": "numbered_section",
                "pattern": r"^\d+\.\s+[A-Z][A-Z\s]+\.?",
                "level": 2,
                "priority": "high",
            },
            # Subsections
            {
                "name": "subsection",
                "pattern": r"^\d+\.\d+\.\s+[A-Z]",
                "level": 3,
                "priority": "medium",
            },
            # Sub-subsections
            {
                "name": "sub_subsection",
                "pattern": r"^\d+\.\d+\.\d+\.\s+[A-Z]",
                "level": 4,
                "priority": "low",
            },
            # Lettered items
            {
                "name": "lettered_item",
                "pattern": r"^[a-z]\.\s+",
                "level": 5,
                "priority": "low",
            },
            # Numbered items
            {
                "name": "numbered_item",
                "pattern": r"^\(\d+\)\s+",
                "level": 6,
                "priority": "none",
            },
        ]

    def detect_structure(self, text: str) -> DocumentStructure:
        """Detect all document structure elements."""
        lines = text.split("\n")

        # Detect sections
        sections = self._detect_sections(lines)

        # Detect tables
        tables = self._detect_tables(lines)

        # Detect paragraphs
        paragraphs = self._detect_paragraphs(lines)

        # Create hierarchy levels
        hierarchy_levels = self._create_hierarchy_levels(sections)

        return DocumentStructure(
            sections=sections,
            tables=tables,
            paragraphs=paragraphs,
            hierarchy_levels=hierarchy_levels,
        )

    def _detect_sections(self, lines: List[str]) -> List[Dict]:
        """Detect section headers and build hierarchical tree in single pass."""
        sections = []
        section_stack = []  # Stack of currently open sections

        for line_num, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue

            # Check if this line is a section header
            section_info = None
            for pattern_info in self.section_patterns:
                if re.match(pattern_info["pattern"], line_stripped, re.IGNORECASE):
                    section_info = pattern_info
                    break

            if section_info:
                # Close sections that are at same or deeper level
                while (
                    section_stack
                    and sections[section_stack[-1]]["level"] >= section_info["level"]
                ):
                    closed_section_idx = section_stack.pop()
                    sections[closed_section_idx]["content_end"] = line_num

                # Create new section
                parent_idx = section_stack[-1] if section_stack else None
                section_idx = len(sections)

                sections.append(
                    {
                        "index": section_idx,
                        "line_number": line_num,
                        "text": line_stripped,
                        "pattern_name": section_info["name"],
                        "level": section_info["level"],
                        "priority": section_info["priority"],
                        "parent": parent_idx,
                        "children": [],
                        "content_start": line_num + 1,
                        "content_end": len(
                            lines
                        ),  # Will be updated when section closes
                        "has_content": False,  # Will be determined later
                    }
                )

                # Add to parent's children
                if parent_idx is not None:
                    sections[parent_idx]["children"].append(section_idx)

                # Push to stack
                section_stack.append(section_idx)

        # Close remaining open sections
        while section_stack:
            closed_section_idx = section_stack.pop()
            sections[closed_section_idx]["content_end"] = len(lines)

        # Determine which sections have actual content (not just subsection headers)
        for section in sections:
            content_lines = lines[section["content_start"] : section["content_end"]]
            # Has content if there are non-empty lines that aren't section headers
            section["has_content"] = any(
                line.strip() and not self._is_section_header(line.strip())
                for line in content_lines
            )

        self.logger.info(f"Detected {len(sections)} hierarchical sections")
        return sections

    def _is_section_header(self, line: str) -> bool:
        """Quick check if a line is a section header."""
        for pattern_info in self.section_patterns:
            if re.match(pattern_info["pattern"], line, re.IGNORECASE):
                return True
        return False

    def _detect_tables(self, lines: List[str]) -> List[Dict]:
        """Detect table boundaries in the text."""
        tables = []
        current_table = None

        for line_num, line in enumerate(lines):
            if "=== TABLE" in line and "START ===" in line:
                if current_table is not None:
                    self.logger.warning("Found table start without previous table end")
                current_table = {
                    "start_line": line_num,
                    "name": line.strip(),
                    "content_start": line_num,
                }
            elif "=== TABLE" in line and "END ===" in line:
                if current_table is not None:
                    current_table["end_line"] = line_num
                    current_table["content_end"] = line_num + 1
                    tables.append(current_table)
                    current_table = None
                else:
                    self.logger.warning("Found table end without matching start")

        if current_table is not None:
            self.logger.warning("Unclosed table section")

        self.logger.info(f"Detected {len(tables)} tables")
        return tables

    def _detect_paragraphs(self, lines: List[str]) -> List[Dict]:
        """Detect paragraph boundaries (groups of lines separated by empty lines)."""
        paragraphs = []
        current_para_start = None

        for line_num, line in enumerate(lines):
            if line.strip():  # Non-empty line
                if current_para_start is None:
                    current_para_start = line_num
            else:  # Empty line
                if current_para_start is not None:
                    paragraphs.append(
                        {
                            "start_line": current_para_start,
                            "end_line": line_num - 1,
                            "content_start": current_para_start,
                            "content_end": line_num,
                        }
                    )
                    current_para_start = None

        # Handle final paragraph
        if current_para_start is not None:
            paragraphs.append(
                {
                    "start_line": current_para_start,
                    "end_line": len(lines) - 1,
                    "content_start": current_para_start,
                    "content_end": len(lines),
                }
            )

        self.logger.info(f"Detected {len(paragraphs)} paragraphs")
        return paragraphs

    def _create_hierarchy_levels(self, sections: List[Dict]) -> Dict[int, List[Dict]]:
        """Group sections by hierarchy level."""
        hierarchy = {}
        for section in sections:
            level = section["level"]
            if level not in hierarchy:
                hierarchy[level] = []
            hierarchy[level].append(section)

        return hierarchy


class UnifiedChunker:
    """Implements the unified chunking decision tree."""

    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.text_cleaner = TextCleaner(logger)
        self.structure_detector = StructureDetector(logger)

    def chunk_text_unified(
        self,
        text: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,  # pylint: disable=unused-argument
        overlap: int = DEFAULT_CHUNK_OVERLAP,  # pylint: disable=unused-argument
    ) -> List[str]:
        """Unified chunking following the decision tree."""
        # Validate configuration
        ConfigurationValidator.validate()

        # Empty text check
        if not text.strip():
            return []

        # Clean text but preserve structure
        clean_text = self.text_cleaner.clean_text_preserve_structure(text)

        # Single chunk check
        if len(clean_text.split()) <= MAX_CHUNK_SIZE:
            return [self.text_cleaner.final_cleanup(clean_text)]

        # Multi-chunk processing
        return self._process_multi_chunk(clean_text)

    def _process_multi_chunk(self, text: str) -> List[str]:
        """Process text that needs multiple chunks using hierarchical structure."""
        # Detect document structure
        structure = self.structure_detector.detect_structure(text)
        lines = text.split("\n")

        if not structure.sections:
            # No structure detected - fallback to paragraph splitting
            return self._fallback_paragraph_splitting(text)

        # Find content-bearing nodes (sections with actual content)
        content_nodes = [s for s in structure.sections if s["has_content"]]

        if not content_nodes:
            # No content-bearing sections - use fallback
            self.logger.info("No content-bearing sections found, using fallback")
            return self._fallback_paragraph_splitting(text)

        # Create chunks by building up from content-bearing nodes
        chunks = []
        processed_sections = set()

        for section in content_nodes:
            if section["index"] in processed_sections:
                continue

            # Build chunk starting from this content node, including ancestors
            chunk_content, covered_sections = self._build_hierarchical_chunk(
                section, structure.sections, lines
            )

            if chunk_content.strip():
                chunks.append(chunk_content)
                processed_sections.update(covered_sections)

        # Handle any remaining unprocessed sections
        for section in structure.sections:
            if section["index"] not in processed_sections and section["has_content"]:
                section_content = self._extract_section_content(
                    section, structure.sections, lines
                )
                if section_content.strip():
                    chunks.append(section_content)

        return self._final_cleanup_and_merge(chunks)

    def _build_hierarchical_chunk(
        self, start_section: Dict, all_sections: List[Dict], lines: List[str]
    ) -> Tuple[str, set]:
        """Build a chunk that includes a section and as many ancestors/siblings as possible."""
        chunk_parts = []
        covered_sections = set()
        current_word_count = 0

        # Start with just this content-bearing section
        chunk_parts.append(start_section["text"])
        covered_sections.add(start_section["index"])

        # Add the section's own content
        section_content = self._extract_section_content(
            start_section, all_sections, lines
        )
        if section_content.strip():
            chunk_parts.append(section_content)

        current_word_count = len("\n".join(chunk_parts).split())

        # Try to add children one by one while staying under size limit
        for child_idx in start_section["children"]:
            child_section = all_sections[child_idx]
            child_size = self._estimate_subtree_size(child_section, all_sections, lines)

            if current_word_count + child_size <= MAX_CHUNK_SIZE:
                # Add this child and its subtree
                chunk_parts.append(child_section["text"])
                covered_sections.add(child_idx)
                self._add_section_tree_to_chunk(
                    child_section, all_sections, lines, chunk_parts, covered_sections
                )
                current_word_count += child_size
            else:
                # Can't fit this child, stop here
                break

        return "\n".join(chunk_parts), covered_sections

    def _find_optimal_root(
        self, section: Dict, all_sections: List[Dict], lines: List[str]
    ) -> Dict:
        """Find the highest ancestor that still keeps chunk under size limit."""
        current = section

        # Walk up the tree while size permits
        while current["parent"] is not None:
            parent = all_sections[current["parent"]]

            # Estimate size if we include the parent's entire subtree
            estimated_size = self._estimate_subtree_size(parent, all_sections, lines)

            if estimated_size <= MAX_CHUNK_SIZE:
                current = parent
            else:
                break

        return current

    def _estimate_subtree_size(
        self, section: Dict, all_sections: List[Dict], lines: List[str]
    ) -> int:
        """Estimate word count for a section and all its descendants."""
        total_words = 0

        # Add section header
        total_words += len(section["text"].split())

        # Add section content
        content = self._extract_section_content(section, all_sections, lines)
        total_words += len(content.split())

        # Recursively add children
        for child_idx in section["children"]:
            child_section = all_sections[child_idx]
            total_words += self._estimate_subtree_size(
                child_section, all_sections, lines
            )

        return total_words

    def _add_section_tree_to_chunk(
        self,
        section: Dict,
        all_sections: List[Dict],
        lines: List[str],
        chunk_parts: List[str],
        covered_sections: set,
    ):
        """Recursively add section and its children to chunk."""
        # Add section content (if any)
        content = self._extract_section_content(section, all_sections, lines)
        if content.strip():
            chunk_parts.append(content)

        # Add children
        for child_idx in section["children"]:
            child_section = all_sections[child_idx]
            if child_idx not in covered_sections:
                chunk_parts.append(child_section["text"])  # Child header
                covered_sections.add(child_idx)
                self._add_section_tree_to_chunk(
                    child_section, all_sections, lines, chunk_parts, covered_sections
                )

    def _extract_section_content(
        self, section: Dict, all_sections: List[Dict], lines: List[str]
    ) -> str:
        """Extract only the direct content of a section (excluding child section content)."""
        content_lines = []

        # Find the line number of the first child section (if any)
        first_child_line = section["content_end"]  # Default to section end
        if section["children"]:
            # Get the minimum line number among all children
            first_child_line = min(
                all_sections[child_idx]["line_number"]
                for child_idx in section["children"]
            )

        # Extract content only up to the first child
        for line_num in range(section["content_start"], first_child_line):
            if line_num < len(lines):
                line = lines[line_num]
                content_lines.append(line)

        return "\n".join(content_lines)

    def _process_section(
        self,
        section: Dict,
        lines: List[str],
        structure: DocumentStructure,
        chunk_size: int,
        overlap: int,
    ) -> List[str]:
        """Process a single top-level section."""
        # Get section content
        section_lines = lines[section["content_start"] : section["content_end"]]
        section_text = "\n".join(section_lines)
        word_count = len(section_text.split())

        if word_count <= TARGET_CHUNK_SIZE:
            # Own chunk
            return [section_text]
        elif word_count <= MAX_CHUNK_SIZE:
            # Try combining with adjacent sections (handled at higher level)
            return [section_text]
        else:
            # Split strategy
            return self._split_oversized_section(
                section, section_lines, structure, chunk_size, overlap
            )

    def _split_oversized_section(
        self,
        section: Dict,
        section_lines: List[str],
        structure: DocumentStructure,
        chunk_size: int,
        overlap: int,
    ) -> List[str]:
        """Split an oversized section using the decision tree."""
        section_text = "\n".join(section_lines)

        # Check for tables in this section
        section_tables = [
            table
            for table in structure.tables
            if (
                section["content_start"] <= table["start_line"] < section["content_end"]
            )
        ]

        if section_tables:
            return self._split_section_with_tables(
                section, section_lines, section_tables, chunk_size, overlap
            )

        # Check for subsections
        subsections = [
            s
            for s in structure.sections
            if (
                s["level"] > section["level"]
                and section["content_start"]
                <= s["line_number"]
                < section["content_end"]
            )
        ]

        if subsections:
            return self._split_section_with_subsections(
                section, section_lines, subsections
            )

        # No clear structure - paragraph/sentence splitting
        return self._split_section_no_structure(section_text)

    def _split_section_with_tables(
        self,
        section: Dict,
        section_lines: List[str],
        tables: List[Dict],
        chunk_size: int,  # pylint: disable=unused-argument
        overlap: int,  # pylint: disable=unused-argument
    ) -> List[str]:
        """Split section that contains tables."""
        chunks = []
        lines = section_lines
        last_pos = 0

        for table in tables:
            # Adjust table positions relative to section start
            table_start = table["start_line"] - section["content_start"]
            table_end = table["end_line"] - section["content_start"]

            # Text before table
            text_before = "\n".join(lines[last_pos:table_start])
            table_text = "\n".join(lines[table_start : table_end + 1])

            # Check if table itself is too large
            table_words = len(table_text.split())
            if table_words > MAX_CHUNK_SIZE:
                # Split large table
                table_chunks = self._split_large_table(table_text)

                # Handle text before
                if text_before.strip():
                    chunks.append(text_before)

                chunks.extend(table_chunks)
            else:
                # Try to combine text before + table
                combined = (
                    text_before + "\n" + table_text
                    if text_before.strip()
                    else table_text
                )
                combined_words = len(combined.split())

                if combined_words <= MAX_CHUNK_SIZE:
                    chunks.append(combined)
                else:
                    # Separate chunks
                    if text_before.strip():
                        chunks.append(text_before)
                    chunks.append(table_text)

            last_pos = table_end + 1

        # Remaining text after last table
        if last_pos < len(lines):
            remaining_text = "\n".join(lines[last_pos:])
            if remaining_text.strip():
                remaining_words = len(remaining_text.split())
                if remaining_words > MAX_CHUNK_SIZE:
                    # Split remaining text
                    remaining_chunks = self._split_section_no_structure(remaining_text)
                    chunks.extend(remaining_chunks)
                else:
                    chunks.append(remaining_text)

        return [chunk for chunk in chunks if chunk.strip()]

    def _split_large_table(self, table_text: str) -> List[str]:
        """Split a table that's larger than MAX_CHUNK_SIZE."""
        lines = table_text.split("\n")
        chunks = []
        current_chunk = []
        current_words = 0

        for line in lines:
            line_words = len(line.split())

            if current_words + line_words > TARGET_CHUNK_SIZE and current_chunk:
                # Finalize current chunk
                chunks.append("\n".join(current_chunk))
                current_chunk = [line]
                current_words = line_words
            else:
                current_chunk.append(line)
                current_words += line_words

        # Add final chunk
        if current_chunk:
            chunks.append("\n".join(current_chunk))

        return chunks

    def _split_section_with_subsections(
        self,
        section: Dict,
        section_lines: List[str],
        subsections: List[Dict],
    ) -> List[str]:
        """Split section using subsection boundaries."""
        chunks = []
        lines = section_lines
        last_pos = 0

        for subsection in subsections:
            # Adjust positions relative to section start
            subsection_start = subsection["line_number"] - section["content_start"]

            # Text before this subsection
            if subsection_start > last_pos:
                text_before = "\n".join(lines[last_pos:subsection_start])
                if text_before.strip():
                    chunks.append(text_before)

            last_pos = subsection_start

        # Handle content from last subsection to end
        if last_pos < len(lines):
            remaining_text = "\n".join(lines[last_pos:])
            if remaining_text.strip():
                chunks.append(remaining_text)

        # Apply merging logic to avoid small chunks
        return self._merge_small_adjacent_chunks(chunks)

    def _split_section_no_structure(self, text: str) -> List[str]:
        """Split section with no clear structure using paragraphs/sentences."""
        # Try paragraph boundaries first
        paragraphs = re.split(r"\n\s*\n", text)

        if len(paragraphs) > 1:
            # Use paragraph boundaries
            chunks = []
            current_chunk = ""
            current_words = 0

            for para in paragraphs:
                para_words = len(para.split())

                if current_words + para_words <= TARGET_CHUNK_SIZE:
                    current_chunk = (
                        current_chunk + "\n\n" + para if current_chunk else para
                    )
                    current_words += para_words
                else:
                    if current_chunk:
                        chunks.append(current_chunk)

                    if para_words > MAX_CHUNK_SIZE:
                        # Split large paragraph at sentences
                        para_chunks = self._split_at_sentences(para)
                        chunks.extend(para_chunks)
                        current_chunk = ""
                        current_words = 0
                    else:
                        current_chunk = para
                        current_words = para_words

            if current_chunk:
                chunks.append(current_chunk)

            return chunks
        else:
            # No paragraphs, split at sentences
            return self._split_at_sentences(text)

    def _split_at_sentences(self, text: str) -> List[str]:
        """Split text at sentence boundaries."""
        # Simple sentence splitting
        sentences = re.split(r"(?<=[.!?])\s+", text)

        chunks = []
        current_chunk = ""
        current_words = 0

        for sentence in sentences:
            sentence_words = len(sentence.split())

            if sentence_words > MAX_CHUNK_SIZE:
                # Split at punctuation near target
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                    current_words = 0

                # Split long sentence at punctuation
                split_chunks = self._split_at_punctuation(sentence)
                chunks.extend(split_chunks)
            elif current_words + sentence_words <= TARGET_CHUNK_SIZE:
                current_chunk = (
                    current_chunk + " " + sentence if current_chunk else sentence
                )
                current_words += sentence_words
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
                current_words = sentence_words

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _split_at_punctuation(self, text: str) -> List[str]:
        """Split text at punctuation marks near TARGET_CHUNK_SIZE."""
        words = text.split()
        chunks = []
        current_chunk = []

        for i, word in enumerate(words):
            current_chunk.append(word)

            if len(current_chunk) >= TARGET_CHUNK_SIZE:
                # Look for punctuation in next few words
                found_punct = False
                for j in range(min(50, len(words) - i)):  # Look ahead max 50 words
                    if i + j < len(words) and re.search(r"[.!?;,]$", words[i + j]):
                        # Include up to punctuation
                        current_chunk.extend(words[i + 1 : i + j + 1])
                        chunks.append(" ".join(current_chunk))
                        current_chunk = []
                        # Skip processed words
                        for _ in range(j + 1):
                            if i + 1 < len(words):
                                i += 1
                        found_punct = True
                        break

                if not found_punct and len(current_chunk) >= MAX_CHUNK_SIZE:
                    # Force split at current position
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _merge_small_adjacent_chunks(self, chunks: List[str]) -> List[str]:
        """Merge small adjacent chunks following decision tree logic."""
        if not chunks:
            return chunks

        merged = []
        current_chunk = ""
        current_words = 0

        for chunk in chunks:
            chunk_words = len(chunk.split())

            if current_words == 0:
                # First chunk
                current_chunk = chunk
                current_words = chunk_words
            elif (current_words < MIN_CHUNK_SIZE or chunk_words < MIN_CHUNK_SIZE) and (
                current_words + chunk_words <= MAX_CHUNK_SIZE
            ):
                # Merge small chunks
                current_chunk = current_chunk + "\n\n" + chunk
                current_words += chunk_words
            else:
                # Finalize current and start new
                merged.append(current_chunk)
                current_chunk = chunk
                current_words = chunk_words

        if current_chunk:
            merged.append(current_chunk)

        return merged

    def _fallback_paragraph_splitting(self, text: str) -> List[str]:
        """Fallback when no structure is detected."""
        self.logger.info("No document structure detected, using paragraph fallback")

        # Split on multiple newlines (paragraph boundaries)
        paragraphs = re.split(r"\n\s*\n\s*\n", text)  # 3+ newlines

        if len(paragraphs) <= 1:
            # No clear paragraphs, try double newlines
            paragraphs = re.split(r"\n\s*\n", text)

        chunks = []
        current_chunk = ""
        current_words = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_words = len(para.split())

            if current_words + para_words <= TARGET_CHUNK_SIZE:
                current_chunk = current_chunk + "\n\n" + para if current_chunk else para
                current_words += para_words
            else:
                if current_chunk:
                    chunks.append(current_chunk)

                if para_words > MAX_CHUNK_SIZE:
                    # Split large paragraph
                    para_chunks = self._split_section_no_structure(para)
                    chunks.extend(para_chunks)
                    current_chunk = ""
                    current_words = 0
                else:
                    current_chunk = para
                    current_words = para_words

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _final_cleanup_and_merge(self, chunks: List[str]) -> List[str]:
        """Final cleanup and merging pass."""
        # Clean whitespace in each chunk
        cleaned_chunks = []
        for chunk in chunks:
            cleaned = self.text_cleaner.final_cleanup(chunk)
            if cleaned:
                cleaned_chunks.append(cleaned)

        # Final merging pass for small adjacent chunks
        merged_chunks = self._merge_small_adjacent_chunks(cleaned_chunks)

        # Enforce size limits - split any oversized chunks
        final_chunks = self._enforce_size_limits(merged_chunks)

        # Log final statistics
        if final_chunks:
            word_counts = [len(chunk.split()) for chunk in final_chunks]
            avg_size = sum(word_counts) / len(word_counts)
            oversized = sum(1 for wc in word_counts if wc > MAX_CHUNK_SIZE)
            undersized = sum(1 for wc in word_counts if wc < MIN_CHUNK_SIZE)

            self.logger.info(
                f"Final chunking result: {len(final_chunks)} chunks, "
                f"avg {avg_size:.0f} words, "
                f"{oversized} oversized, {undersized} undersized"
            )

        return final_chunks

    def _enforce_size_limits(self, chunks: List[str]) -> List[str]:
        """Emergency splitting for any oversized chunks."""
        final_chunks = []

        for chunk in chunks:
            word_count = len(chunk.split())
            if word_count <= MAX_CHUNK_SIZE:
                final_chunks.append(chunk)
            else:
                # Split oversized chunk progressively
                self.logger.info(f"Splitting oversized chunk: {word_count} words")
                split_chunks = self._progressive_split(chunk)
                final_chunks.extend(split_chunks)

        return final_chunks

    def _progressive_split(self, text: str) -> List[str]:
        """Split oversized text using progressive strategy."""
        # Target slightly smaller than TARGET_CHUNK_SIZE for safety
        target_size = int(TARGET_CHUNK_SIZE * 0.9)  # 630 words instead of 700

        # Try splitting on paragraph boundaries first
        paragraphs = re.split(r"\n\s*\n", text)
        if len(paragraphs) > 1:
            chunks = self._build_chunks_from_parts(paragraphs, target_size)
            if all(len(chunk.split()) <= MAX_CHUNK_SIZE for chunk in chunks):
                return chunks

        # Try splitting on sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if len(sentences) > 1:
            chunks = self._build_chunks_from_parts(sentences, target_size)
            if all(len(chunk.split()) <= MAX_CHUNK_SIZE for chunk in chunks):
                return chunks

        # Last resort: split at punctuation marks
        parts = re.split(r"([.!?;:,])\s*", text)
        # Rejoin punctuation with preceding text
        rejoined_parts = []
        for i in range(0, len(parts) - 1, 2):
            if i + 1 < len(parts):
                rejoined_parts.append(parts[i] + parts[i + 1])
            else:
                rejoined_parts.append(parts[i])

        return self._build_chunks_from_parts(rejoined_parts, target_size)

    def _build_chunks_from_parts(self, parts: List[str], target_size: int) -> List[str]:
        """Build chunks from text parts while staying under target size."""
        chunks = []
        current_chunk = ""
        current_words = 0

        for part in parts:
            part_words = len(part.split())

            # If adding this part would exceed MAX_CHUNK_SIZE, finalize current chunk
            if current_words + part_words > MAX_CHUNK_SIZE:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = part
                current_words = part_words
            # If adding this part would exceed target but we're under max, decide based on size
            elif current_words + part_words > target_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = part
                current_words = part_words
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += " " + part
                else:
                    current_chunk = part
                current_words += part_words

        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        return [chunk for chunk in chunks if chunk.strip()]


# Legacy compatibility classes and functions
class PDFExtractor:
    """Legacy PDF extraction for compatibility."""

    def __init__(self, logger=None, quiet_mode=False):
        self.logger = logger or logging.getLogger(__name__)
        self.quiet_mode = quiet_mode

    def extract_text(self, pdf_path: str) -> str:
        """Extract text from PDF - simplified for new implementation."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                all_text = []

                for page_num, page in enumerate(pdf.pages):
                    # Extract tables
                    tables = page.find_tables()

                    # Format tables
                    for table_num, table in enumerate(tables):
                        try:
                            raw_data = table.extract()
                            if raw_data and any(
                                any(cell for cell in row if cell) for row in raw_data
                            ):
                                formatted_table = self._format_table(
                                    raw_data, page_num, table_num
                                )
                                if formatted_table:
                                    all_text.append(formatted_table)
                        except Exception as e:
                            self.logger.warning(
                                f"Error extracting table {table_num} from page {page_num}: {e}"
                            )

                    # Extract page text
                    page_text = page.extract_text()
                    if page_text:
                        all_text.append(page_text)

                return "\n\n".join(all_text)

        except Exception as e:
            self.logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""

    def _format_table(self, table_data, page_num: int, table_num: int) -> str:
        """Format table data for chunking."""
        if not table_data:
            return ""

        formatted_lines = [f"=== TABLE {page_num + 1}.{table_num + 1} START ==="]

        for row in table_data:
            if not row:
                continue

            row_parts = []
            for cell in row:
                if cell is not None:
                    clean_cell = str(cell).strip().replace("\n", " | ")
                    if clean_cell:
                        row_parts.append(clean_cell)

            if row_parts:
                formatted_lines.append(" || ".join(row_parts))

        formatted_lines.append(f"=== TABLE {page_num + 1}.{table_num + 1} END ===")
        return "\n".join(formatted_lines)


class DocumentProcessor:
    """Main document processor - drop-in replacement for the old one."""

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        enable_header_footer_removal: bool = True,
    ):
        self.logger = logging.getLogger(__name__)
        self.quiet_mode = False

        # Initialize components
        self.pdf_extractor = PDFExtractor(self.logger, self.quiet_mode)
        self.chunker = UnifiedChunker(self.logger)
        self.embedding_manager = EmbeddingManager(
            embedding_model, self.logger, self.quiet_mode
        )

        # For compatibility
        self.enable_header_footer_removal = enable_header_footer_removal

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file."""
        return self.pdf_extractor.extract_text(pdf_path)

    def clean_text(self, text: str) -> str:
        """Clean extracted text."""
        return self.chunker.text_cleaner.clean_text_preserve_structure(text)

    def chunk_text(
        self,
        text: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> List[str]:
        """Chunk text using the unified algorithm."""
        return self.chunker.chunk_text_unified(text, chunk_size, overlap)

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for text chunks."""
        return self.embedding_manager.generate_embeddings(texts)

    def create_embeddings_for_docs(
        self, documents: List[Dict]
    ) -> Tuple[List[str], np.ndarray, List[Dict]]:
        """Create embeddings for all document chunks."""
        return self.embedding_manager.create_embeddings_for_docs(documents)

    def process_document(self, pdf_path: str, quiet: bool = False) -> Dict:
        """Process a single PDF document."""
        filename = os.path.basename(pdf_path)
        if not quiet:
            self.logger.info(f"Processing: {filename}")

        # Extract text
        raw_text = self.extract_text_from_pdf(pdf_path)
        if not raw_text.strip():
            self.logger.warning(f"No text extracted from {filename}")
            return None

        # Clean text
        clean_text = self.clean_text(raw_text)

        # Create chunks
        chunks = self.chunk_text(clean_text)
        if not chunks:
            self.logger.warning(f"No chunks created from {filename}")
            return None

        # Extract DOD document number
        first_page = clean_text[:FIRST_PAGE_SAMPLE_CHARS] if clean_text else None
        doc_number = extract_dod_number(filename, first_page_text=first_page)

        return {
            "filename": filename,
            "path": pdf_path,
            "raw_text": raw_text,
            "clean_text": clean_text,
            "chunks": chunks,
            "chunk_count": len(chunks),
            "doc_number": doc_number,
        }

    def process_directory(
        self, directory: str, show_dir_info: bool = True, verbose: bool = False
    ) -> List[Dict]:
        """Process all PDF files in directory."""
        pdf_files = list(Path(directory).glob("*.pdf"))
        if show_dir_info:
            print(f"Found {len(pdf_files)} PDF files in {directory}")

        processed_docs = []

        for i, pdf_path in enumerate(pdf_files, 1):
            if verbose:
                print(
                    f"\rProcessing {i}/{len(pdf_files)}: {pdf_path.name:<50}",
                    end="",
                    flush=True,
                )

            doc_data = self.process_document(str(pdf_path), quiet=True)
            if doc_data:
                processed_docs.append(doc_data)

        if verbose:
            print()  # New line after progress

        self.logger.info(f"Successfully processed {len(processed_docs)} documents")
        return processed_docs


class EmbeddingManager:
    """Embedding manager for compatibility."""

    def __init__(self, model_name="all-MiniLM-L6-v2", logger=None, quiet_mode=False):
        self.model_name = model_name
        self.model = None
        self.logger = logger or logging.getLogger(__name__)
        self.quiet_mode = quiet_mode

    def _load_embedding_model(self):
        """Lazy load embedding model."""
        if self.model is None:
            if not self.quiet_mode:
                self.logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
        return self.model

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for text chunks."""
        model = self._load_embedding_model()
        return model.encode(texts, show_progress_bar=not self.quiet_mode)

    def create_embeddings_for_docs(
        self, documents: List[Dict]
    ) -> Tuple[List[str], np.ndarray, List[Dict]]:
        """Create embeddings for all document chunks."""
        all_chunks = []
        chunk_metadata = []

        for doc in documents:
            for i, chunk in enumerate(doc["chunks"]):
                all_chunks.append(chunk)
                chunk_metadata.append(
                    {
                        "doc_filename": doc["filename"],
                        "doc_path": doc["path"],
                        "chunk_index": i,
                        "chunk_text": chunk,
                        "doc_number": doc.get("doc_number"),
                    }
                )

        if not self.quiet_mode:
            self.logger.info(
                f"Generating embeddings for {len(all_chunks)} total chunks..."
            )

        embeddings = self.generate_embeddings(all_chunks)
        return all_chunks, embeddings, chunk_metadata


# Legacy functions for compatibility
def save_processed_documents(documents: List[Dict], output_file: str) -> None:
    """Save processed documents to JSON file."""
    docs_to_save = []
    for doc in documents:
        doc_copy = doc.copy()
        doc_copy.pop("raw_text", None)  # Remove raw_text to save space
        docs_to_save.append(doc_copy)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(docs_to_save, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(docs_to_save)} processed documents to {output_file}")


def load_processed_documents(input_file: str) -> List[Dict]:
    """Load processed documents from JSON file."""
    with open(input_file, "r", encoding="utf-8") as f:
        documents = json.load(f)

    print(f"Loaded {len(documents)} processed documents from {input_file}")
    return documents
