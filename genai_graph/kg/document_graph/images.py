"""Image extraction and caption parsing for Document Graph sections."""

from __future__ import annotations

import re
from pathlib import Path

from loguru import logger

from genai_graph.kg.nodes.document_section import Image, MarkdownSection

# Matches: <!-- Image: filename.png (hash: 1234abcd) -->
# optionally followed by markdown image: ![alt](url "title")
_MISTRAL_IMG_COMMENT_PATTERN = re.compile(
    r"<!--\s*Image:\s*(?P<filename>[^\s\(\)]+)\s*\(hash:\s*(?P<hash>[a-fA-F0-9]+)\)\s*-->",
    re.IGNORECASE,
)

# Matches standard markdown image syntax: ![alt](url) or ![alt](url "title")
_MD_IMAGE_PATTERN = re.compile(r"!\[(?P<alt>.*?)\]\((?P<url>[^\s\)\"\']+)(?:\s+[\"'](?P<title>.*?)[\"'])?\)")

# Matches caption patterns like "Fig. 1: ...", "Figure 2 - ...", "*Figure 1: ...*", "_Source: ..._"
_CAPTION_PREFIX_PATTERN = re.compile(
    r"^(?:[\*_]{1,2})?\s*(?:(?:Fig(?:ure)?\.?|Chart|Diagram|Graph|Illustration|Photo|Table|Source)\s*[\d\w\.\-:]*|Source\s*:)\s*",
    re.IGNORECASE,
)

_ITALIC_BOLD_WRAPPER = re.compile(r"^[\*_]{1,2}(.*?)[\*_]{1,2}$")
_HTML_CAPTION_PATTERN = re.compile(r"<figcaption>(.*?)</figcaption>|<p[^>]*><em>(.*?)</em></p>", re.IGNORECASE)


def _clean_caption_text(text: str) -> str:
    """Clean markdown styling and HTML tags from a caption string."""
    text = text.strip()
    html_match = _HTML_CAPTION_PATTERN.search(text)
    if html_match:
        text = html_match.group(1) or html_match.group(2) or text

    italic_match = _ITALIC_BOLD_WRAPPER.match(text)
    if italic_match:
        text = italic_match.group(1)

    # Strip residual formatting
    text = re.sub(r"[\*_`]", "", text)
    return text.strip()


def _is_generic_alt(alt: str, filename: str, img_hash: str) -> bool:
    """Check if alt text is generic/useless placeholder."""
    cleaned = alt.strip().lower()
    if not cleaned:
        return True
    if cleaned in ("image", "img", "figure", "photo", "picture", "chart", "illustration"):
        return True
    if cleaned.startswith("image:") or cleaned.startswith("img_") or cleaned == filename.lower():
        return True
    if cleaned == img_hash.lower():
        return True
    return False


def _find_caption_after_pos(text: str, pos: int) -> str | None:
    """Scan lines following an image tag to extract an immediate caption/figure description."""
    sub = text[pos:]
    lines = sub.splitlines()
    # Scan up to 4 non-empty lines
    non_empty_count = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Stop if we hit a heading or another image or HTML block
        if stripped.startswith("#") or stripped.startswith("![") or stripped.startswith("<!--"):
            break

        non_empty_count += 1
        if non_empty_count > 4:
            break

        # Check for explicit caption prefix or italic line
        is_caption_prefix = bool(_CAPTION_PREFIX_PATTERN.match(stripped))
        is_italic_or_html = (
            (stripped.startswith("*") and stripped.endswith("*"))
            or (stripped.startswith("_") and stripped.endswith("_"))
            or "<figcaption>" in stripped
            or "<em>" in stripped
        )

        if is_caption_prefix or is_italic_or_html:
            cleaned = _clean_caption_text(stripped)
            if len(cleaned) >= 3:
                return cleaned

    return None


def extract_section_images(
    section: MarkdownSection,
    markdown_file_path: Path | None = None,
) -> list[Image]:
    """Extract all images referenced in a MarkdownSection with metadata and captions.

    Args:
        section: The MarkdownSection to parse.
        markdown_file_path: Path to the containing markdown file (for resolving relative paths).

    Returns:
        List of Image nodes found within the section.
    """
    text = section.text or ""
    if not text or ("![" not in text and "<!-- Image:" not in text):
        return []

    images: list[Image] = []
    seen_hashes: set[str] = set()

    # Strategy 1: Look for combined Mistral OCR comments + markdown links
    # e.g.:
    # <!-- Image: filename.png (hash: 1234abcd) -->
    # ![alt](path/to/filename.png)
    # *Fig. 1: Caption*
    pos = 0
    for comm_match in _MISTRAL_IMG_COMMENT_PATTERN.finditer(text):
        fn = comm_match.group("filename")
        h = comm_match.group("hash").lower()
        comm_end = comm_match.end()

        # Look for the corresponding ![alt](url) right after the comment
        after_comm = text[comm_end : comm_end + 500]
        md_match = _MD_IMAGE_PATTERN.search(after_comm)

        alt_text = ""
        url = fn
        tag_end = comm_end
        if md_match and md_match.start() < 100:  # within 100 chars of comment
            alt_text = md_match.group("alt")
            url = md_match.group("url")
            tag_end = comm_end + md_match.end()

        caption = _find_caption_after_pos(text, tag_end)
        if not caption and not _is_generic_alt(alt_text, fn, h):
            caption = _clean_caption_text(alt_text)

        img_obj = _build_image_node(
            section=section,
            image_hash=h,
            filename=fn or Path(url).name,
            url=url,
            caption=caption,
            markdown_file_path=markdown_file_path,
        )
        if img_obj.image_hash not in seen_hashes:
            seen_hashes.add(img_obj.image_hash)
            images.append(img_obj)

    # Strategy 2: Look for any standalone markdown images ![alt](url) not already captured
    for md_match in _MD_IMAGE_PATTERN.finditer(text):
        url = md_match.group("url")
        alt_text = md_match.group("alt")
        tag_end = md_match.end()

        # Derive hash from filename stem or xxhash
        stem = Path(url).stem.lower()
        # If stem looks like hex hash (8-64 chars), use it; otherwise use stem
        h = stem

        if h in seen_hashes:
            continue

        caption = _find_caption_after_pos(text, tag_end)
        if not caption and not _is_generic_alt(alt_text, Path(url).name, h):
            caption = _clean_caption_text(alt_text)

        img_obj = _build_image_node(
            section=section,
            image_hash=h,
            filename=Path(url).name,
            url=url,
            caption=caption,
            markdown_file_path=markdown_file_path,
        )
        if img_obj.image_hash not in seen_hashes:
            seen_hashes.add(img_obj.image_hash)
            images.append(img_obj)

    return images


def _build_image_node(
    section: MarkdownSection,
    image_hash: str,
    filename: str,
    url: str,
    caption: str | None,
    markdown_file_path: Path | None,
) -> Image:
    """Resolve file path, compute size, and instantiate an Image node."""
    resolved_path: Path | None = None
    file_size: int | None = None

    url_path = Path(url)
    if url_path.is_absolute() and url_path.exists():
        resolved_path = url_path
    elif markdown_file_path is not None:
        # Check relative to markdown file parent
        candidate1 = markdown_file_path.parent / url_path
        # Check relative to markdown file parent / images
        candidate2 = markdown_file_path.parent / "images" / filename
        # Check relative to CWD / data
        candidate3 = Path.cwd() / url_path

        for cand in (candidate1, candidate2, candidate3):
            if cand.exists():
                resolved_path = cand
                break

    if resolved_path is None:
        # Fallback check against CWD
        cand = Path.cwd() / url_path
        if cand.exists():
            resolved_path = cand

    stored_path = url
    if resolved_path is not None:
        try:
            file_size = resolved_path.stat().st_size
            # Make relative to CWD if possible
            try:
                stored_path = str(resolved_path.relative_to(Path.cwd()))
            except ValueError:
                stored_path = str(resolved_path)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not stat image {}: {}", resolved_path, exc)

    return Image(
        image_id=f"{section.section_id}::{image_hash}",
        section_id=section.section_id,
        markdown_hash=section.markdown_hash,
        image_hash=image_hash,
        name=image_hash,
        filename=filename,
        path=stored_path,
        description=caption,
        size=file_size,
    )
