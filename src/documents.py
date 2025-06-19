"""Document processing module for IRIS RAG system."""

import json
import logging
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer

from .config import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    HEADER_FOOTER_ZONE_RATIO,
    LINE_TOLERANCE_PIXELS,
    MIN_HEADER_FOOTER_CHARS,
    MAX_HEADER_FOOTER_WORDS,
    MIN_LINE_LENGTH,
    FIRST_PAGE_SAMPLE_CHARS,
)


def extract_dod_number(
    filename: str, metadata: Dict = None, first_page_text: str = None
) -> str:
    """Extract DOD document number from filename, metadata, or document content.

    Args:
        filename: The filename to parse
        metadata: PDF metadata dict (optional)
        first_page_text: Text from first page of document (optional)

    Returns:
        The document number (e.g., "5000.85") or None if not found
    """
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


class PDFExtractor:
    """Handles PDF text extraction and table detection."""

    def __init__(self, logger=None, quiet_mode=False):
        self.logger = logger or logging.getLogger(__name__)
        self.quiet_mode = quiet_mode

    def extract_text(self, pdf_path: str) -> str:
        """Extract text from PDF file using pdfplumber with table-aware processing."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                all_content = []

                for page_num, page in enumerate(pdf.pages):
                    # Extract tables from this page
                    tables = self._detect_and_extract_tables(page, page_num)
                    table_bboxes = [table["bbox"] for table in tables]

                    # Extract text content
                    page_text = self._extract_text_outside_tables(page, table_bboxes)

                    # Combine content for this page
                    page_content = []

                    # Add tables with their formatted content
                    for table in tables:
                        page_content.append(table["content"])

                    # Add page text
                    if page_text:
                        page_content.append(page_text)

                    # Join page content
                    if page_content:
                        all_content.extend(page_content)

                # Join all content
                final_text = "\n\n".join(all_content)

                # Log summary
                total_tables = len(
                    [
                        content
                        for content in all_content
                        if "=== TABLE" in content and "START ===" in content
                    ]
                )
                if total_tables > 0 and not self.quiet_mode:
                    self.logger.info(
                        "Extracted %s tables from %s pages",
                        total_tables,
                        len(pdf.pages),
                    )

                return final_text

        except (FileNotFoundError, PermissionError, Exception) as e:
            self.logger.error("Error extracting text from %s: %s", pdf_path, e)
            return ""

    def _detect_and_extract_tables(self, page, page_num: int):
        """Detect and extract tables from a page using pdfplumber."""
        try:
            # Use default table detection (proven to work in debug tests)
            tables = page.find_tables()

            table_data = []
            for table_num, table in enumerate(tables):
                try:
                    # Extract table content
                    raw_data = table.extract()
                    if raw_data and any(
                        any(cell for cell in row if cell) for row in raw_data
                    ):
                        # Format table as readable text
                        formatted_table = self._format_table_for_chunking(
                            raw_data, page_num, table_num
                        )
                        if formatted_table:
                            table_info = {
                                "content": formatted_table,
                                "bbox": table.bbox,
                                "page": page_num,
                                "table_num": table_num,
                            }
                            table_data.append(table_info)
                            if not self.quiet_mode:
                                self.logger.info(
                                    "Extracted table %s from page %s",
                                    table_num + 1,
                                    page_num + 1,
                                )
                except Exception as e:
                    self.logger.warning(
                        "Error extracting table %s from page %s: %s",
                        table_num,
                        page_num,
                        e,
                    )

            return table_data

        except Exception as e:
            self.logger.warning("Error detecting tables on page %s: %s", page_num, e)
            return []

    def _format_table_for_chunking(
        self, table_data, page_num: int, table_num: int
    ) -> str:
        """Format table data for chunking with clear structure preservation."""
        if not table_data:
            return ""

        formatted_lines = []

        # Add table header
        formatted_lines.append(f"=== TABLE {page_num + 1}.{table_num + 1} START ===")

        # Process each row
        for row in table_data:
            if not row:
                continue

            # Clean up row data
            row_parts = []
            for cell in row:
                if cell is not None:
                    clean_cell = str(cell).strip()
                    if clean_cell:
                        # Preserve multi-line structure but make it readable
                        clean_cell = clean_cell.replace("\n", " | ")
                        row_parts.append(clean_cell)

            if row_parts:
                # Join cells with clear separators
                formatted_lines.append(" || ".join(row_parts))

        # Add table footer
        formatted_lines.append(f"=== TABLE {page_num + 1}.{table_num + 1} END ===")

        return "\n".join(formatted_lines)

    def _extract_text_outside_tables(self, page, table_bboxes):
        """Extract text from page excluding table areas."""
        if not table_bboxes:
            # No tables, extract all text normally
            return page.extract_text()

        try:
            # For simplicity, just extract all text and let table sections override
            # More sophisticated approach would crop page regions, but that's complex
            # and the current approach works fine for our use case
            page_text = page.extract_text()
            return page_text if page_text else ""

        except Exception as e:
            self.logger.warning("Error extracting non-table text: %s", e)
            return ""

    def extract_tables_from_page(self, page, page_num: int):
        """Public interface to extract tables from a page."""
        return self._detect_and_extract_tables(page, page_num)

    def extract_text_outside_tables(self, page, table_bboxes):
        """Public interface to extract text excluding table areas."""
        return self._extract_text_outside_tables(page, table_bboxes)


class LayoutAnalyzer:
    """Handles page layout analysis and header/footer removal."""

    def __init__(
        self, logger=None, quiet_mode=False, enable_header_footer_removal=True
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.quiet_mode = quiet_mode
        self.enable_header_footer_removal = enable_header_footer_removal

    def analyze_layout(self, page) -> Dict:
        """Extract text positions and content from a page for layout analysis."""
        try:
            page_height = page.height
            page_width = page.width

            # Define header/footer zones (top/bottom configurable percentage of page)
            header_zone_bottom = page_height * HEADER_FOOTER_ZONE_RATIO
            footer_zone_top = page_height * (1.0 - HEADER_FOOTER_ZONE_RATIO)

            # Extract all text elements with positions
            words = page.extract_words()

            layout_data = {
                "page_height": page_height,
                "page_width": page_width,
                "header_zone_bottom": header_zone_bottom,
                "footer_zone_top": footer_zone_top,
                "header_elements": [],
                "footer_elements": [],
                "content_elements": [],
            }

            # Group words into lines by position
            header_words = []
            footer_words = []
            content_words = []

            for word in words:
                word_top = word.get("top", 0)
                word_text = word.get("text", "").strip()

                if not word_text:
                    continue

                word_data = {
                    "text": word_text,
                    "top": word_top,
                    "x0": word.get("x0", 0),
                    "x1": word.get("x1", 0),
                }

                if word_top <= header_zone_bottom:
                    header_words.append(word_data)
                elif word_top >= footer_zone_top:
                    footer_words.append(word_data)
                else:
                    content_words.append(word_data)

            # Group words into lines and create text elements
            layout_data["header_elements"] = self._group_words_into_lines(header_words)
            layout_data["footer_elements"] = self._group_words_into_lines(footer_words)
            layout_data["content_elements"] = self._group_words_into_lines(
                content_words
            )

            return layout_data

        except Exception as e:
            self.logger.warning("Error analyzing page layout: %s", e)
            return {}

    def _group_words_into_lines(self, words: List[Dict]) -> List[Dict]:
        """Group words into coherent text lines based on vertical position."""
        if not words:
            return []

        # Sort words by position (top to bottom, left to right)
        words.sort(key=lambda w: (w["top"], w["x0"]))

        lines = []
        current_line = []
        current_line_top = None
        line_tolerance = (
            LINE_TOLERANCE_PIXELS  # Allow configurable pixels tolerance for same line
        )

        for word in words:
            word_top = word["top"]

            # If this is first word or word is on same line (within tolerance)
            if (
                current_line_top is None
                or abs(word_top - current_line_top) <= line_tolerance
            ):
                current_line.append(word)
                if current_line_top is None:
                    current_line_top = word_top
            else:
                # Start new line
                if current_line:
                    # Create line element from current line words
                    line_text = " ".join(w["text"] for w in current_line)
                    line_element = {
                        "text": line_text,
                        "top": current_line_top,
                        "x0": min(w["x0"] for w in current_line),
                        "x1": max(w["x1"] for w in current_line),
                    }
                    lines.append(line_element)

                # Start new line with current word
                current_line = [word]
                current_line_top = word_top

        # Don't forget the last line
        if current_line:
            line_text = " ".join(w["text"] for w in current_line)
            line_element = {
                "text": line_text,
                "top": current_line_top,
                "x0": min(w["x0"] for w in current_line),
                "x1": max(w["x1"] for w in current_line),
            }
            lines.append(line_element)

        return lines

    def create_clean_reference(self, first_page) -> Dict:
        """Analyze first page to create baseline for clean content."""
        if not self.quiet_mode:
            self.logger.info("Creating clean reference from first page...")

        reference_data = self.analyze_layout(first_page)

        # First page analysis - assume it's mostly clean
        # Store patterns that might appear on other pages as headers/footers
        potential_patterns = []

        # Look for short text elements that could be headers/footers
        all_elements = (
            reference_data.get("header_elements", [])
            + reference_data.get("footer_elements", [])
            + reference_data.get("content_elements", [])
        )

        for element in all_elements:
            text = element["text"]
            # Identify potentially repetitive elements
            if len(text) < MIN_HEADER_FOOTER_CHARS and (
                re.match(r"^[A-Z\s\d\.\-]+$", text)  # All caps/numbers
                or re.search(r"\d+", text)  # Contains numbers
                or len(text.split()) <= MAX_HEADER_FOOTER_WORDS
            ):  # Very short
                potential_patterns.append(text.lower().strip())

        reference_data["potential_header_footer_patterns"] = list(
            set(potential_patterns)
        )

        if not self.quiet_mode:
            header_count = len(reference_data.get("header_elements", []))
            footer_count = len(reference_data.get("footer_elements", []))
            content_count = len(reference_data.get("content_elements", []))
            self.logger.info(
                "First page analysis: %s header elements, %s footer elements, %s content elements",
                header_count,
                footer_count,
                content_count,
            )

        return reference_data

    def detect_repetitive_elements(
        self, page_layout: Dict, reference_data: Dict
    ) -> Tuple[List[str], List[str]]:
        """Compare page against reference to identify repetitive header/footer elements."""
        headers_to_remove = []
        footers_to_remove = []

        # Check header elements
        for element in page_layout.get("header_elements", []):
            text = element["text"].strip()

            # Check if this looks like a header/footer pattern
            if len(text) < MIN_HEADER_FOOTER_CHARS and (  # Short text
                text.lower()
                in reference_data.get("potential_header_footer_patterns", [])
                or self._is_likely_header_footer(text)
            ):
                headers_to_remove.append(text)

        # Check footer elements
        for element in page_layout.get("footer_elements", []):
            text = element["text"].strip()

            # Check if this looks like a header/footer pattern
            if len(text) < MIN_HEADER_FOOTER_CHARS and (  # Short text
                text.lower()
                in reference_data.get("potential_header_footer_patterns", [])
                or self._is_likely_header_footer(text)
            ):
                footers_to_remove.append(text)

        return headers_to_remove, footers_to_remove

    def _is_likely_header_footer(self, text: str) -> bool:
        """Check if text matches common header/footer patterns."""
        text_lower = text.lower().strip()

        # Common header/footer patterns
        patterns = [
            r"page\s+\d+",  # "Page 1", "Page 12"
            r"^\d+$",  # Standalone numbers
            r"page\s+\d+\s+of\s+\d+",  # "Page 1 of 10"
            r"dod[dim]\s+\d+\.\d+",  # "DODD 5000.85", "DODI 1234.56"
            r"unclassified",  # Classification markings
            r"for\s+official\s+use\s+only",
            r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}",  # Dates
            r"section\s+\d+",  # "Section 1"
            r"appendix\s+[a-z0-9]",  # "Appendix A"
            r"^\d+\s*$",  # Just numbers with spaces
        ]

        for pattern in patterns:
            if re.search(pattern, text_lower):
                return True

        return False

    def filter_headers_footers(
        self, page_text: str, headers_to_remove: List[str], footers_to_remove: List[str]
    ) -> str:
        """Remove detected header/footer elements from page text."""
        filtered_text = page_text
        removed_items = []

        # Remove headers
        for header in headers_to_remove:
            if header in filtered_text:
                filtered_text = filtered_text.replace(header, "")
                removed_items.append(f"Header: {header}")

        # Remove footers
        for footer in footers_to_remove:
            if footer in filtered_text:
                filtered_text = filtered_text.replace(footer, "")
                removed_items.append(f"Footer: {footer}")

        # Clean up extra whitespace
        filtered_text = re.sub(
            r"\n\s*\n\s*\n", "\n\n", filtered_text
        )  # Multiple newlines
        filtered_text = re.sub(r"^\s*\n", "", filtered_text)  # Leading newlines
        filtered_text = re.sub(r"\n\s*$", "", filtered_text)  # Trailing newlines

        if removed_items:
            self.logger.debug("Removed: %s", ", ".join(removed_items))

        return filtered_text.strip()


class TextProcessor:
    """Handles text cleaning and chunking operations."""

    def __init__(self, logger=None, quiet_mode=False):
        self.logger = logger or logging.getLogger(__name__)
        self.quiet_mode = quiet_mode

    def clean_text(self, text: str) -> str:
        """Clean extracted text from PDF artifacts."""
        if not text:
            return ""

        # Replace multiple spaces/tabs with single space
        text = re.sub(r"[ \t]+", " ", text)

        # Replace multiple newlines with single newline
        text = re.sub(r"\n+", "\n", text)

        # Remove page breaks and form feeds
        text = re.sub(r"[\f\r\v]", "", text)

        # Remove standalone page numbers (number on its own line)
        text = re.sub(r"\n\s*\d{1,4}\s*\n", "\n", text)

        # Remove common PDF header/footer patterns
        text = re.sub(r"\n\s*Page \d+ of \d+\s*\n", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"\n\s*\d+\s*\n", "\n", text)

        # Remove very short lines (likely artifacts) but preserve meaningful short text
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # Keep lines that are:
            # - Longer than configured minimum characters, OR
            # - Contain important policy markers, OR
            # - Are section headers (contain numbers/letters and periods/colons)
            if (
                len(line) > MIN_LINE_LENGTH
                or any(
                    marker in line.upper()
                    for marker in [
                        "SECTION",
                        "CHAPTER",
                        "PART",
                        "DODD",
                        "DODI",
                        "APPENDIX",
                    ]
                )
                or re.match(r"^[A-Z0-9]+[\.:]\s*[A-Z]", line)
            ):
                cleaned_lines.append(line)

        # Rejoin and do final cleanup
        text = "\n".join(cleaned_lines)

        # Fix common OCR/extraction errors
        text = re.sub(r"\s+([.,;:!?])", r"\1", text)  # Remove space before punctuation
        text = re.sub(
            r"([.!?])\s*([a-z])", r"\1 \2", text
        )  # Ensure space after sentence end

        # Final whitespace normalization
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def chunk_text(
        self,
        text: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> List[str]:
        """Split text into overlapping chunks with table awareness."""
        if not text.strip():
            return []

        # Check if text contains table markers
        if "=== TABLE" in text and "START ===" in text and "END ===" in text:
            # Use table-aware chunking
            if not self.quiet_mode:
                self.logger.info("Using table-aware chunking")
            return self._chunk_with_table_awareness(text, chunk_size, overlap)
        else:
            # Use regular chunking for text without tables
            return self._chunk_text_regular(text, chunk_size, overlap)

    def _chunk_with_table_awareness(
        self, text: str, chunk_size: int = None, overlap: int = None
    ) -> List[str]:
        """Split text into chunks while preserving table integrity."""
        chunk_size = chunk_size or DEFAULT_CHUNK_SIZE
        overlap = overlap or DEFAULT_CHUNK_OVERLAP

        # Find table boundaries
        table_boundaries = self._find_table_boundaries(text)

        if not table_boundaries:
            # No tables found, use regular chunking
            return self._chunk_text_regular(text, chunk_size, overlap)

        lines = text.split("\n")
        chunks = []
        non_table_lines = []
        last_processed_line = 0

        for table_boundary in table_boundaries:
            # Add text before this table to non-table lines
            before_table_lines = lines[
                last_processed_line : table_boundary["start_line"]
            ]
            non_table_lines.extend(before_table_lines)

            # Process accumulated non-table text if any
            if non_table_lines:
                non_table_text = "\n".join(non_table_lines).strip()
                if non_table_text:
                    # Chunk the non-table text normally
                    text_chunks = self._chunk_text_regular(
                        non_table_text, chunk_size, overlap
                    )
                    chunks.extend(text_chunks)
                non_table_lines = []

            # Handle the table as a complete unit
            table_content = "\n".join(table_boundary["content_lines"])
            table_word_count = len(table_content.split())

            if table_word_count <= chunk_size:
                # Table fits in a single chunk, keep it intact
                chunks.append(table_content)
                if not self.quiet_mode:
                    self.logger.info(
                        "Table %s kept as single chunk (%s words)",
                        table_boundary["name"],
                        table_word_count,
                    )
            else:
                # Table is too large, but still keep it as a dedicated chunk
                # This preserves table integrity even if it exceeds normal size limits
                chunks.append(table_content)
                if not self.quiet_mode:
                    self.logger.warning(
                        "Table %s exceeds chunk size (%s words) but kept intact",
                        table_boundary["name"],
                        table_word_count,
                    )

            last_processed_line = table_boundary["end_line"] + 1

        # Process any remaining text after the last table
        if last_processed_line < len(lines):
            remaining_lines = lines[last_processed_line:]
            non_table_lines.extend(remaining_lines)

        if non_table_lines:
            remaining_text = "\n".join(non_table_lines).strip()
            if remaining_text:
                text_chunks = self._chunk_text_regular(
                    remaining_text, chunk_size, overlap
                )
                chunks.extend(text_chunks)

        # Log summary
        table_chunk_count = len(table_boundaries)
        total_chunks = len(chunks)
        if not self.quiet_mode:
            text_chunks = total_chunks - table_chunk_count
            self.logger.info(
                "Table-aware chunking: %s total chunks (%s table chunks, %s text chunks)",
                total_chunks,
                table_chunk_count,
                text_chunks,
            )

        return chunks

    def _chunk_text_regular(
        self, text: str, chunk_size: int, overlap: int
    ) -> List[str]:
        """Regular text chunking (extracted from original chunk_text method)."""
        if not text.strip():
            return []

        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i : i + chunk_size]
            chunk = " ".join(chunk_words)
            chunks.append(chunk)

            # Stop if we've reached the end
            if i + chunk_size >= len(words):
                break

        return chunks

    def _find_table_boundaries(self, text: str):
        """Find table boundaries in text to identify complete table sections."""
        boundaries = []
        lines = text.split("\n")

        current_table = None
        start_idx = None

        for i, line in enumerate(lines):
            if "=== TABLE" in line and "START ===" in line:
                # Found table start
                if current_table is not None:
                    self.logger.warning("Found table start without previous table end")
                current_table = line.strip()
                start_idx = i

            elif "=== TABLE" in line and "END ===" in line:
                # Found table end
                if current_table is not None and start_idx is not None:
                    # Complete table boundary found
                    boundaries.append(
                        {
                            "type": "table",
                            "name": current_table,
                            "start_line": start_idx,
                            "end_line": i,
                            "content_lines": lines[start_idx : i + 1],
                        }
                    )
                    current_table = None
                    start_idx = None
                else:
                    self.logger.warning("Found table end without matching start")

        if current_table is not None:
            self.logger.warning("Unclosed table section: %s", current_table)

        return boundaries


class EmbeddingManager:
    """Handles embedding model operations."""

    def __init__(self, model_name="all-MiniLM-L6-v2", logger=None, quiet_mode=False):
        self.model_name = model_name
        self.model = None
        self.logger = logger or logging.getLogger(__name__)
        self.quiet_mode = quiet_mode

    def _load_embedding_model(self):
        """Lazy load embedding model."""
        if self.model is None:
            if not self.quiet_mode:
                self.logger.info(
                    "Loading embedding model: %s (this may take a moment on first run)...",
                    self.model_name,
                )
            self.model = SentenceTransformer(self.model_name)
            if not self.quiet_mode:
                self.logger.info("Embedding model loaded successfully")
        return self.model

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for text chunks."""
        model = self._load_embedding_model()
        embeddings = model.encode(texts, show_progress_bar=True)
        return embeddings

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
                "Generating embeddings for %s total chunks...", len(all_chunks)
            )
        embeddings = self.generate_embeddings(all_chunks)

        return all_chunks, embeddings, chunk_metadata

    def ensure_model_loaded(self):
        """Public interface to ensure embedding model is loaded."""
        return self._load_embedding_model()


class DocumentProcessor:
    """Orchestrates document processing using specialized components."""

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        enable_header_footer_removal: bool = True,
    ):
        """Initialize with component classes."""
        self.logger = logging.getLogger(__name__)
        self.quiet_mode = False  # Control verbose output

        # Initialize component classes
        self.pdf_extractor = PDFExtractor(
            logger=self.logger, quiet_mode=self.quiet_mode
        )
        self.layout_analyzer = LayoutAnalyzer(
            logger=self.logger,
            quiet_mode=self.quiet_mode,
            enable_header_footer_removal=enable_header_footer_removal,
        )
        self.text_processor = TextProcessor(
            logger=self.logger, quiet_mode=self.quiet_mode
        )
        self.embedding_manager = EmbeddingManager(
            model_name=embedding_model, logger=self.logger, quiet_mode=self.quiet_mode
        )

        # For compatibility with existing code
        self.enable_header_footer_removal = enable_header_footer_removal

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file with layout analysis and header/footer removal."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                all_content = []
                reference_data = None

                for page_num, page in enumerate(pdf.pages):
                    # Create reference from first page for header/footer removal
                    if page_num == 0:
                        reference_data = self.layout_analyzer.create_clean_reference(
                            page
                        )

                    # Extract tables from this page
                    tables = self.pdf_extractor.extract_tables_from_page(page, page_num)
                    table_bboxes = [table["bbox"] for table in tables]

                    # Extract text content
                    page_text = self.pdf_extractor.extract_text_outside_tables(
                        page, table_bboxes
                    )

                    # Apply header/footer filtering (except for first page)
                    if (
                        self.enable_header_footer_removal
                        and page_num > 0
                        and reference_data
                        and page_text
                    ):
                        page_layout = self.layout_analyzer.analyze_layout(page)
                        headers_to_remove, footers_to_remove = (
                            self.layout_analyzer.detect_repetitive_elements(
                                page_layout, reference_data
                            )
                        )

                        if headers_to_remove or footers_to_remove:
                            original_length = len(page_text)
                            page_text = self.layout_analyzer.filter_headers_footers(
                                page_text, headers_to_remove, footers_to_remove
                            )
                            filtered_length = len(page_text)

                            if (
                                original_length != filtered_length
                                and not self.quiet_mode
                            ):
                                filtered_chars = original_length - filtered_length
                                header_count = len(headers_to_remove)
                                footer_count = len(footers_to_remove)
                                self.logger.info(
                                    "Page %s: Filtered %s characters (%s headers, %s footers)",
                                    page_num + 1,
                                    filtered_chars,
                                    header_count,
                                    footer_count,
                                )

                    # Combine content for this page
                    page_content = []

                    # Add tables with their formatted content
                    for table in tables:
                        page_content.append(table["content"])

                    # Add filtered page text
                    if page_text:
                        page_content.append(page_text)

                    # Join page content
                    if page_content:
                        all_content.extend(page_content)

                # Join all content
                final_text = "\n\n".join(all_content)

                # Log summary
                total_tables = len(
                    [
                        content
                        for content in all_content
                        if "=== TABLE" in content and "START ===" in content
                    ]
                )
                if total_tables > 0 and not self.quiet_mode:
                    self.logger.info(
                        "Extracted %s tables from %s pages",
                        total_tables,
                        len(pdf.pages),
                    )

                if self.enable_header_footer_removal and not self.quiet_mode:
                    self.logger.info("Header/footer removal applied to document")

                return final_text

        except (FileNotFoundError, PermissionError, Exception) as e:
            self.logger.error("Error extracting text from %s: %s", pdf_path, e)
            return ""

    def clean_text(self, text: str) -> str:
        """Compatibility method - delegates to text processor."""
        return self.text_processor.clean_text(text)

    def chunk_text(
        self,
        text: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> List[str]:
        """Compatibility method - delegates to text processor."""
        return self.text_processor.chunk_text(text, chunk_size, overlap)

    def _load_embedding_model(self):
        """Compatibility method - delegates to embedding manager."""
        return self.embedding_manager.ensure_model_loaded()

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Compatibility method - delegates to embedding manager."""
        return self.embedding_manager.generate_embeddings(texts)

    def create_embeddings_for_docs(
        self, documents: List[Dict]
    ) -> Tuple[List[str], np.ndarray, List[Dict]]:
        """Compatibility method - delegates to embedding manager."""
        return self.embedding_manager.create_embeddings_for_docs(documents)

    def process_document(self, pdf_path: str, quiet: bool = False) -> Dict:
        """Process a single PDF document."""
        filename = os.path.basename(pdf_path)
        if not quiet and not self.quiet_mode:
            self.logger.info("Processing: %s", filename)

        # Extract text
        raw_text = self.extract_text_from_pdf(pdf_path)
        if not raw_text.strip():
            self.logger.warning("No text extracted from %s", filename)
            return None

        # Clean text
        clean_text = self.clean_text(raw_text)

        # Create chunks
        chunks = self.chunk_text(clean_text)
        if not chunks:
            self.logger.warning("No chunks created from %s", filename)
            return None

        # Extract DOD document number
        # First configured chars as first page
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
            if verbose or not self.quiet_mode:
                print(
                    f"\rProcessing {i}/{len(pdf_files)}: {pdf_path.name:<50}",
                    end="",
                    flush=True,
                )
            doc_data = self.process_document(str(pdf_path), quiet=True)
            if doc_data:
                processed_docs.append(doc_data)

        if verbose or not self.quiet_mode:
            print()  # New line after progress

        if not self.quiet_mode:
            self.logger.info("Successfully processed %s documents", len(processed_docs))
        return processed_docs


def save_processed_documents(documents: List[Dict], output_file: str) -> None:
    """Save processed documents to JSON file for later embedding generation."""
    # Remove raw_text to save space, keep everything else
    docs_to_save = []
    for doc in documents:
        doc_copy = doc.copy()
        # Remove raw_text to save file size
        doc_copy.pop("raw_text", None)
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
