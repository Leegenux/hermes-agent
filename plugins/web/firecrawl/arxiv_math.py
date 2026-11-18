"""Single-source TeX for arXiv HTML returned by Firecrawl; no extra fetches."""
import re
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit


def is_arxiv_html(url: str) -> bool:
    try:
        parsed = urlsplit(url)
        return (parsed.scheme == 'https' and parsed.netloc == 'arxiv.org' and not parsed.query
                and re.fullmatch(r'/html/\d{4}\.\d{4,5}(?:v[1-9]\d*)?/?', parsed.path) is not None)
    except ValueError:
        return False


def scrape_formats(url: str, formats: list[str]) -> list[str]:
    return formats + ['rawHtml'] if is_arxiv_html(url) and 'markdown' in formats and 'rawHtml' not in formats else formats


def _compact(text: str) -> str:
    return re.sub(r'\s+', '', text)


class _Article(HTMLParser):
    def __init__(self, url: str):
        super().__init__(convert_charrefs=True)
        self.url = url
        self.active = False
        self.articles = 0
        self.math = None
        self.annotation = False
        self.ignored = None
        self.parts = []
        self.prose = []
        self.formulas = []
        self.links = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'article':
            if self.active:
                raise ValueError('nested article')
            self.active = True
            self.articles += 1
        if not self.active:
            return
        if self.math is not None:
            if tag == 'math':
                raise ValueError('nested math')
            if tag == 'annotation' and attrs.get('encoding') == 'application/x-tex':
                self.annotation = True
            return
        if tag == 'math':
            if not attrs.get('alttext'):
                raise ValueError('missing TeX source')
            self.math = dict(tex=attrs['alttext'], annotation='', display=attrs.get('display'))
            self.parts.append(' ⟪FORMULA_' + str(len(self.formulas)) + '⟫ ')
            return
        self._start_content(tag, attrs)

    def _start_content(self, tag, attrs):
        if tag in ('script', 'style'):
            self.ignored = tag
        if self.ignored:
            return
        if tag == 'pre':
            # Preserve code indentation via the original provider rather than flattening it.
            raise ValueError('preformatted content requires original rendering')
        if tag in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6'):
            self.parts.append('\n\n' + '#' * int(tag[1]) + ' ')
        elif tag in ('p', 'div', 'section', 'figure', 'figcaption', 'li', 'tr', 'table', 'br'):
            self.parts.append('\n')
        if tag == 'td':
            for key in ('colspan', 'rowspan'):
                if attrs.get(key):
                    self.parts.append(' [' + key + '=' + attrs[key] + '] ')
        if tag == 'a':
            href = attrs.get('href', '')
            target = href if href.startswith('#') else urljoin(self.url, href)
            valid = bool(href) and (target.startswith('#') or urlsplit(target).scheme in ('http', 'https'))
            self.links.append(target if valid else None)
            if valid:
                self.parts.append('[')
        if tag == 'img':
            src = urljoin(self.url, attrs.get('src', ''))
            if urlsplit(src).scheme in ('http', 'https'):
                self.parts.append(' [IMAGE: ' + src + ' ' + attrs.get('alt', '') + '] ')

    def handle_endtag(self, tag):
        if not self.active:
            return
        if self.math is not None:
            if tag == 'annotation':
                self.annotation = False
            if tag == 'math':
                if _compact(self.math['tex']) != _compact(self.math['annotation']):
                    raise ValueError('disagreeing TeX sources')
                self.formulas.append(self.math)
                self.math = None
            return
        if tag == self.ignored:
            self.ignored = None
        if self.ignored:
            return
        if tag == 'a' and self.links:
            target = self.links.pop()
            if target:
                self.parts.append('](' + target + ')')
        if tag in ('p', 'div', 'section', 'figure', 'figcaption', 'li', 'tr', 'table',
                   'h1', 'h2', 'h3', 'h4', 'h5', 'h6'):
            self.parts.append('\n')
        if tag == 'td':
            self.parts.append(' | ')
        if tag == 'article':
            self.active = False

    def handle_data(self, data):
        if not self.active or self.ignored:
            return
        if self.math is not None:
            if self.annotation:
                self.math['annotation'] += data
        else:
            self.parts.append(data)
            if data.strip():
                self.prose.append(data)

    def render(self) -> str:
        if self.articles != 1 or self.active or self.math is not None or not self.formulas or self.links:
            raise ValueError('incomplete article')
        text = re.sub(r'[ \t\f\v]+', ' ', ''.join(self.parts))
        text = re.sub(r' *\n *', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text).strip()
        for index, formula in enumerate(self.formulas):
            marker = '⟪FORMULA_' + str(index) + '⟫'
            if text.count(marker) != 1:
                raise ValueError('ambiguous formula placeholder')
            delimiter = '$$' if formula['display'] == 'block' else '$'
            text = text.replace(marker, delimiter + formula['tex'] + delimiter)
        pos, searchable = 0, _compact(text)
        for chunk in self.prose:
            needle = _compact(chunk)
            found = searchable.find(needle, pos)
            if found < 0:
                raise ValueError('prose sequence not preserved')
            pos = found + len(needle)
        return text + '\n'


def normalize_arxiv(url: str, payload: dict, fallback: str) -> str:
    if not is_arxiv_html(url):
        return fallback
    metadata = payload.get('metadata')
    final_url = metadata.get('sourceURL', url) if isinstance(metadata, dict) else url
    if not isinstance(final_url, str) or final_url.split('#', 1)[0] != url.split('#', 1)[0]:
        return fallback
    html = payload.get('rawHtml') or payload.get('raw_html')
    # Bound parser work; retain the provider's original result for oversized or missing HTML.
    if not isinstance(html, str) or not html or len(html) > 4_000_000:
        return fallback
    try:
        article = _Article(url)
        article.feed(html)
        article.close()
        return article.render()
    except (ValueError, TypeError):
        return fallback
