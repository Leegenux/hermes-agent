"""Behavior contracts for single-source TeX through both Firecrawl entry points."""
import asyncio
import unittest
from unittest.mock import Mock, patch

from plugins.web.firecrawl import provider
from plugins.web import keyless_mcp

URL = 'https://arxiv.org/html/1711.05957v1'
HTML = '''<article><h1>Paper</h1><p>Before
<math alttext="x^2"><semantics><msup><mi>x</mi><mn>2</mn></msup>
<annotation encoding="application/x-tex">x^2</annotation></semantics></math>
after <a href="#ref">citation</a>.</p><figure><img src="plot.png" alt="Plot">
<figcaption>Figure caption.</figcaption></figure></article>'''
FALLBACK = 'Before x2x^2 after citation. Figure caption.'


class TestArxivMath(unittest.TestCase):
    def payload(self, html=HTML):
        return {'markdown': FALLBACK, 'rawHtml': html,
                'metadata': {'title': 'Paper', 'sourceURL': URL}}

    def direct(self, payload=None, url=URL, format=None):
        client = Mock()
        client.scrape.return_value = {'data': self.payload() if payload is None else payload}
        with patch.object(provider, '_get_firecrawl_client', return_value=client), \
             patch.object(provider, 'is_safe_url', return_value=True), \
             patch.object(provider, 'check_website_access', return_value=None):
            result = asyncio.run(provider._scrape_one(url, [format] if format else ['markdown', 'html'], format))
        return result, client

    def test_direct_preserves_single_tex_and_document_content(self):
        result, client = self.direct()
        self.assertIn('$x^2$', result['content'])
        self.assertEqual(result['content'].count('x^2'), 1)
        for text in ['Paper', 'Before', 'after', 'citation', 'Figure caption.', '#ref', 'plot.png']:
            self.assertIn(text, result['content'])
        self.assertEqual(result['raw_content'], result['content'])
        self.assertEqual(result['title'], 'Paper')
        self.assertIn('rawHtml', client.scrape.call_args.kwargs['formats'])

    def test_keyless_uses_the_same_conversion_and_single_scrape(self):
        client = Mock()
        client.scrape.return_value = {'data': self.payload()}
        with patch.object(provider, '_KeylessFirecrawlClient', return_value=client):
            results = keyless_mcp.firecrawl_extract_keyless([URL])
        self.assertIn('$x^2$', results[0]['content'])
        self.assertEqual(client.scrape.call_count, 1)
        self.assertIn('rawHtml', client.scrape.call_args.kwargs['formats'])

    def test_keyless_mixed_batch_leaves_other_page_unchanged(self):
        client = Mock()
        other = 'https://example.com/page'
        client.scrape.side_effect = lambda **kwargs: {'data': {
            **self.payload(), 'metadata': {'sourceURL': kwargs['url']}}}
        with patch.object(provider, '_KeylessFirecrawlClient', return_value=client):
            results = keyless_mcp.firecrawl_extract_keyless([URL, other])
        self.assertEqual([r['url'] for r in results], [URL, other])
        self.assertIn('$x^2$', results[0]['content'])
        self.assertEqual(results[1]['content'], FALLBACK)
        self.assertEqual(client.scrape.call_args_list[1].kwargs['formats'], ['markdown'])
        self.assertEqual(client.scrape.call_count, 2)

    def test_preformatted_code_uses_original_rendering(self):
        result, _ = self.direct(self.payload(HTML.replace('<p>', '<pre>')))
        self.assertEqual(result['content'], FALLBACK)

    def test_missing_or_disagreeing_formula_source_preserves_original(self):
        bad = [HTML.replace('alttext="x^2"', ''),
               HTML.replace('>x^2</annotation>', '>y</annotation>'),
               HTML.replace('<annotation encoding="application/x-tex">x^2</annotation>', ''),
               HTML.replace('</article>', ''), '<html>not an article</html>',
               HTML.replace('</article>', '</article><article>other</article>')]
        for html in bad:
            with self.subTest(html=html):
                result, _ = self.direct(self.payload(html))
                self.assertEqual(result['content'], FALLBACK)

    def test_missing_raw_html_preserves_original(self):
        result, _ = self.direct({'markdown': FALLBACK, 'metadata': {'sourceURL': URL}})
        self.assertEqual(result['content'], FALLBACK)

    def test_sdk_snake_case_raw_html(self):
        data = self.payload()
        data['raw_html'] = data.pop('rawHtml')
        result, _ = self.direct(data, format='markdown')
        self.assertIn('$x^2$', result['content'])

    def test_other_urls_keep_formats_and_output_unchanged(self):
        urls = ['https://example.com/html/1711.05957v1', 'https://arxiv.org.evil.test/html/1711.05957v1',
                'https://arxiv.org/abs/1711.05957v1', 'https://arxiv.org/pdf/1711.05957v1',
                'https://arxiv.org/html/not-a-paper', 'https://arxiv.org/html/1711.05957v0',
                'https://arxiv.org/html/1711.05957v1?other=1']
        for url in urls:
            with self.subTest(url=url):
                result, client = self.direct(url=url)
                self.assertEqual(result['content'], FALLBACK)
                self.assertEqual(client.scrape.call_args.kwargs['formats'], ['markdown', 'html'])

    def test_explicit_html_is_not_normalized(self):
        data = self.payload()
        data['html'] = HTML
        result, client = self.direct(data, format='html')
        self.assertEqual(result['content'], HTML)
        self.assertEqual(client.scrape.call_args.kwargs['formats'], ['html'])

    def test_redirect_identity_change_keeps_original(self):
        data = self.payload()
        data['metadata']['sourceURL'] = 'https://example.com/paper'
        result, _ = self.direct(data)
        self.assertEqual(result['content'], FALLBACK)

    def test_policy_block_remains_before_network(self):
        with patch.object(provider, '_get_firecrawl_client') as network, \
             patch.object(provider, 'check_website_access', return_value=dict(
                 message='blocked by fixture', host='arxiv.org', rule='deny', source='test')):
            result = asyncio.run(provider._scrape_one(URL, ['markdown'], None))
        self.assertEqual(result['error'], 'blocked by fixture')
        network.assert_not_called()


if __name__ == '__main__':
    unittest.main()
