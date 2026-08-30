const assert = require('node:assert/strict');
const path = require('node:path');
const test = require('node:test');
const {
    escapeJsonForHtml,
    normalizePostId,
    normalizeSlug,
    resolvePathInside,
    sanitizeRenderedHtml,
} = require('../scripts/create-post-from-markdown');

const ROOT = path.resolve(__dirname, '..');

test('path segments accept expected values and reject traversal', () => {
    assert.equal(normalizeSlug('safe-post_2'), 'safe-post_2');
    assert.equal(normalizePostId('11407'), '11407');
    assert.throws(() => normalizeSlug('../index'), /slug/i);
    assert.throws(() => normalizePostId('..'), /id/i);
    assert.throws(() => resolvePathInside(path.join(ROOT, 'posts'), '..', 'index.html'), /outside/i);
    assert.match(resolvePathInside(path.join(ROOT, 'posts'), 'safe-post.html'), /safe-post\.html$/);
});

test('rendered Markdown removes executable markup and unsafe URLs', () => {
    const sanitized = sanitizeRenderedHtml(`
        <p onclick="alert(1)">safe <a href="javascript:alert(1)">link</a></p>
        <script>alert(1)</script><img src="data:text/html,boom" onerror="alert(1)">
    `);

    assert.doesNotMatch(sanitized, /script|onclick|onerror|javascript:|data:text/i);
    assert.match(sanitized, /<p>safe/);
});

test('metadata JSON cannot close its script element', () => {
    const escaped = escapeJsonForHtml('{"title":"</script><script>alert(1)</script>"}');
    assert.doesNotMatch(escaped, /<\/script>/i);
    assert.match(escaped, /\\u003c\/script\\u003e/i);
});
