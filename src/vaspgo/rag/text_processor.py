"""
Text processing utilities using markitdown for converting various file formats to Markdown.
"""
from pathlib import Path
from typing import Union

from markitdown import MarkItDown


def convert_file_to_markdown(file_path: Union[str, Path]) -> str:
    """
    Convert various file formats (HTML, PDF, XLSX, DOCX, PPTX, etc.) to Markdown using markitdown.
    
    Supported formats:
    - HTML/HTM: Web pages and HTML files
    - PDF: PDF documents
    - XLSX/XLS: Excel spreadsheets
    - DOCX/DOC: Word documents
    - PPTX/PPT: PowerPoint presentations
    - CSV: Comma-separated values
    - JSON: JSON data files
    - XML: XML documents
    - MD/TXT: Markdown and plain text files (passed through)
    
    Args:
        file_path: Path to the file to convert (can be string or Path object)
        
    Returns:
        Markdown string content
        
    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If conversion fails
        
    Example:
        >>> md_content = convert_file_to_markdown("document.pdf")
        >>> print(md_content)
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        markitdown = MarkItDown()
        result = markitdown.convert(str(file_path))
        return result.text_content
    except Exception as e:
        raise ValueError(f"Failed to convert {file_path} to markdown: {str(e)}")


def convert_url_to_markdown(url: str) -> str:
    """
    Convert a URL (HTML page) to Markdown using markitdown.
    
    Args:
        url: URL to convert (must start with http:// or https://)
        
    Returns:
        Markdown string content
        
    Raises:
        ValueError: If URL is invalid or conversion fails
        
    Example:
        >>> md_content = convert_url_to_markdown("https://www.example.com/page.html")
        >>> print(md_content)
    """
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"Invalid URL: {url}. Must start with http:// or https://")
    
    try:
        markitdown = MarkItDown()
        result = markitdown.convert(url)
        return result.text_content
    except Exception as e:
        raise ValueError(f"Failed to convert URL {url} to markdown: {str(e)}")


def is_supported_format(file_path: Union[str, Path]) -> bool:
    """
    Check if file format is supported by markitdown.
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        True if format is supported, False otherwise
    """
    file_path = Path(file_path)
    ext = file_path.suffix.lower()
    
    supported_extensions = {
        '.html', '.htm',      # HTML files
        '.pdf',               # PDF documents
        '.xlsx', '.xls',      # Excel spreadsheets
        '.docx', '.doc',      # Word documents
        '.pptx', '.ppt',      # PowerPoint presentations
        '.md',                # Markdown files
        '.txt',               # Plain text files
        '.csv',               # CSV files
        '.json',              # JSON files
        '.xml',               # XML files
    }
    
    return ext in supported_extensions


__all__ = [
    "convert_file_to_markdown",
    "convert_url_to_markdown",
    "is_supported_format",
]
