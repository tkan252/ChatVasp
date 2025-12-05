import asyncio
import re
from pathlib import Path
from typing import Dict, List

import aiofiles
import aiohttp
from trafilatura import fetch_url, extract

from autogen_core.memory import Memory, MemoryContent, MemoryMimeType

from vaspgo.rag.text_processor import (
    convert_file_to_markdown,
    convert_url_to_markdown,
    is_supported_format,
)


class DocumentIndexer:
    """Basic document indexer for AutoGen Memory."""

    def __init__(
        self, 
        memory: Memory, 
        chunk_size: int = 1500,
        chunk_tokens: int = 1000,
        overlap_tokens: int = 200,
        use_smart_chunking: bool = True
    ) -> None:
        """
        Initialize DocumentIndexer.
        
        Args:
            memory: Memory instance for storing indexed content
            chunk_size: Legacy parameter for character-based chunking (deprecated, use chunk_tokens)
            chunk_tokens: Target token count per chunk for smart chunking
            overlap_tokens: Token count for overlap between chunks
            use_smart_chunking: Whether to use smart chunking based on Markdown headings
        """
        self.memory = memory
        self.chunk_size = chunk_size  # Keep for backward compatibility
        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.use_smart_chunking = use_smart_chunking

    async def _fetch_url_content(self, url: str) -> str:
        """
        Fetch and extract URL content using trafilatura.
        Falls back to markitdown, then HTML stripping if needed.
        """
        # Try trafilatura first (best for web content extraction)
        try:
            downloaded = fetch_url(url)
            result = extract(downloaded, output_format="markdown")
            if result:
                return result
        except Exception:
            pass
        
        # Fallback 1: Try markitdown conversion
        try:
            return convert_url_to_markdown(url)
        except Exception:
            pass
        
        # Fallback 2: Use aiohttp with HTML stripping
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status()
                content_type = response.headers.get('Content-Type', '').lower()
                
                if 'text/html' in content_type or 'application/xhtml' in content_type:
                    html_content = await response.text()
                    return self._strip_html(html_content)
                else:
                    return await response.text()
    
    async def _fetch_url_with_retry(self, url: str) -> str:
        """
        Fetch URL content with retry logic (up to 3 attempts).
        """
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                return await self._fetch_url_content(url)
            except (aiohttp.ClientError, aiohttp.ServerError, asyncio.TimeoutError) as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Request failed for {url}, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"Request failed for {url} after {max_retries} attempts")
                    raise
            except Exception as e:
                # Non-network errors: don't retry
                raise

    async def _fetch_content(self, source: str) -> str:
        """
        Fetch content from URL or file.
        Automatically converts HTML, PDF, XLSX, etc. to Markdown using markitdown.
        For URLs, retries up to 3 times on failure.
        """
        if source.startswith(("http://", "https://")):
            return await self._fetch_url_with_retry(source)
        else:
            # Handle file paths
            source_path = Path(source)
            if not source_path.is_absolute():
                # If relative path, resolve relative to project root
                project_root = Path(__file__).parent.parent.parent
                source_path = project_root / source_path
            
            # Check if file exists
            if not source_path.exists():
                raise FileNotFoundError(f"File not found: {source_path}")
            
            # Check if it's a supported format that should be converted
            if is_supported_format(source_path):
                # Use markitdown to convert to markdown
                # Note: convert_file_to_markdown is synchronous, but we're in async context
                # This is fine as markitdown operations are typically fast
                return convert_file_to_markdown(source_path)
            else:
                # For unsupported formats or plain text files, read directly
                async with aiofiles.open(source_path, "r", encoding="utf-8") as f:
                    content = await f.read()
                    # If content looks like HTML, try to convert it anyway
                    if "<" in content and ">" in content:
                        html_pattern = re.compile(r'<[^>]+>', re.IGNORECASE)
                        if html_pattern.search(content):
                            try:
                                return convert_file_to_markdown(source_path)
                            except Exception:
                                # If conversion fails, use fallback HTML stripping
                                return self._strip_html(content)
                    return content

    def _strip_html(self, html_content: str) -> str:
        """
        Fallback method to strip HTML tags if markitdown is not available or fails.
        This is a simple implementation for basic HTML content.
        """
        # Remove script and style elements
        html_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        html_content = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        # Remove HTML tags
        html_content = re.sub(r'<[^>]+>', '', html_content)
        # Decode HTML entities (basic)
        html_content = html_content.replace('&nbsp;', ' ')
        html_content = html_content.replace('&amp;', '&')
        html_content = html_content.replace('&lt;', '<')
        html_content = html_content.replace('&gt;', '>')
        html_content = html_content.replace('&quot;', '"')
        # Clean up whitespace
        html_content = re.sub(r'\s+', ' ', html_content)
        return html_content.strip()

    def _approx_token_len(self, text: str) -> int:
        """
        Approximate token count for English text.
        For English, roughly 1 token ≈ 4 characters.
        """
        if not text:
            return 0
        # Simple approximation: ~4 characters per token for English
        return len(text) // 4
    
    def _split_paragraphs_with_headings(self, text: str) -> List[Dict]:
        """
        Split text into paragraphs based on Markdown heading hierarchy, maintaining semantic integrity.
        
        Returns:
            List of paragraph dictionaries with content, heading_path, start, and end positions
        """
        lines = text.splitlines()
        heading_stack: List[str] = []
        paragraphs: List[Dict] = []
        buf: List[str] = []
        char_pos = 0
        
        def flush_buf(end_pos: int):
            """Flush current buffer to paragraphs."""
            if not buf:
                return
            content = "\n".join(buf).strip()
            if not content:
                return
            paragraphs.append({
                "content": content,
                "heading_path": " > ".join(heading_stack) if heading_stack else None,
                "start": max(0, end_pos - len(content)),
                "end": end_pos,
            })
        
        for ln in lines:
            raw = ln
            stripped = raw.strip()
            
            # Check if line is a Markdown heading
            if stripped.startswith("#"):
                # Process heading line
                flush_buf(char_pos)
                
                # Count heading level (# = 1, ## = 2, etc.)
                level = len(stripped) - len(stripped.lstrip('#'))
                title = stripped.lstrip('#').strip()
                
                if level <= 0:
                    level = 1
                
                # Update heading stack based on level
                if level <= len(heading_stack):
                    heading_stack = heading_stack[:level - 1]
                heading_stack.append(title)
                
                char_pos += len(raw) + 1
                continue
            
            # Accumulate paragraph content
            if stripped == "":
                flush_buf(char_pos)
                buf = []
            else:
                buf.append(raw)
            
            char_pos += len(raw) + 1
        
        # Flush remaining buffer
        flush_buf(char_pos)
        
        # If no paragraphs found, return entire text as one paragraph
        if not paragraphs:
            paragraphs = [{
                "content": text,
                "heading_path": None,
                "start": 0,
                "end": len(text)
            }]
        
        return paragraphs
    
    def _chunk_paragraphs(
        self, 
        paragraphs: List[Dict], 
        chunk_tokens: int, 
        overlap_tokens: int
    ) -> List[Dict]:
        """
        Intelligent chunking based on token count, maintaining paragraph boundaries.
        
        Args:
            paragraphs: List of paragraph dictionaries from _split_paragraphs_with_headings
            chunk_tokens: Target token count per chunk
            overlap_tokens: Token count for overlap between chunks
            
        Returns:
            List of chunk dictionaries with content, start, end, and heading_path
        """
        chunks: List[Dict] = []
        cur: List[Dict] = []
        cur_tokens = 0
        i = 0
        
        while i < len(paragraphs):
            p = paragraphs[i]
            p_tokens = self._approx_token_len(p["content"]) or 1
            
            # Add paragraph if it fits or if current chunk is empty
            if cur_tokens + p_tokens <= chunk_tokens or not cur:
                cur.append(p)
                cur_tokens += p_tokens
                i += 1
            else:
                # Generate current chunk
                content = "\n\n".join(x["content"] for x in cur)
                start = cur[0]["start"]
                end = cur[-1]["end"]
                # Get the most recent heading path from current paragraphs
                heading_path = next(
                    (x["heading_path"] for x in reversed(cur) if x.get("heading_path")),
                    None
                )
                
                chunks.append({
                    "content": content,
                    "start": start,
                    "end": end,
                    "heading_path": heading_path,
                })
                
                # Build overlap section
                if overlap_tokens > 0 and cur:
                    kept: List[Dict] = []
                    kept_tokens = 0
                    # Keep paragraphs from the end for overlap
                    for x in reversed(cur):
                        t = self._approx_token_len(x["content"]) or 1
                        if kept_tokens + t > overlap_tokens:
                            break
                        kept.append(x)
                        kept_tokens += t
                    cur = list(reversed(kept))
                    cur_tokens = kept_tokens
                else:
                    cur = []
                    cur_tokens = 0
        
        # Process remaining chunk
        if cur:
            content = "\n\n".join(x["content"] for x in cur)
            start = cur[0]["start"]
            end = cur[-1]["end"]
            heading_path = next(
                (x["heading_path"] for x in reversed(cur) if x.get("heading_path")),
                None
            )
            
            chunks.append({
                "content": content,
                "start": start,
                "end": end,
                "heading_path": heading_path,
            })
        
        return chunks
    
    def _split_text(self, text: str) -> List[Dict]:
        """
        Split text into chunks using smart chunking if enabled, otherwise use simple chunking.
        
        Returns:
            List of chunk dictionaries with 'content' and optionally 'heading_path'
        """
        if self.use_smart_chunking:
            # Use smart chunking based on Markdown headings and token count
            paragraphs = self._split_paragraphs_with_headings(text)
            chunks_dict = self._chunk_paragraphs(paragraphs, self.chunk_tokens, self.overlap_tokens)
            return chunks_dict
        else:
            # Fallback to simple character-based chunking
            chunks: List[Dict] = []
            paragraphs = text.split("\n\n")
            current_chunk = ""
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                    
                # If current chunk plus new paragraph exceeds size, save current chunk
                if current_chunk and len(current_chunk) + len(para) + 2 > self.chunk_size:
                    chunks.append({"content": current_chunk.strip()})
                    current_chunk = para
                else:
                    if current_chunk:
                        current_chunk += "\n\n" + para
                    else:
                        current_chunk = para
            
            # If remaining text is still too large, force split
            if len(current_chunk) > self.chunk_size:
                for i in range(0, len(current_chunk), self.chunk_size):
                    chunk = current_chunk[i : i + self.chunk_size]
                    chunks.append({"content": chunk.strip()})
            elif current_chunk:
                chunks.append({"content": current_chunk.strip()})
            
            return chunks

    async def index_documents(self, sources: List[str]) -> int:
        """
        Index documents into memory.
        Automatically converts HTML, PDF, XLSX, etc. to Markdown before indexing.
        """
        total_chunks = 0
        for source in sources:
            try:
                # _fetch_content now handles conversion automatically
                content = await self._fetch_content(source)
                
                # Split content into chunks
                chunks = self._split_text(content)
                
                # Add chunks to memory
                for i, chunk_dict in enumerate(chunks):
                    metadata = {
                        "source": source,
                        "chunk_index": i
                    }
                    # Add heading_path if available (from smart chunking)
                    if chunk_dict.get("heading_path"):
                        metadata["heading_path"] = chunk_dict["heading_path"]
                    
                    await self.memory.add(
                        MemoryContent(
                            content=chunk_dict["content"],
                            mime_type=MemoryMimeType.TEXT,
                            metadata=metadata
                        )
                    )
                total_chunks += len(chunks)
                print(f"Indexed {len(chunks)} chunks from {source}")
            except Exception as e:
                print(f"Error indexing {source}: {str(e)}")
        
        return total_chunks

__all__ = ["DocumentIndexer"]

async def main():
    memory = None
    indexer = DocumentIndexer(memory)
    sources = ["https://www.vasp.at/wiki/ISTART"]
    await indexer.index_documents(sources)
    print("Indexing complete")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
