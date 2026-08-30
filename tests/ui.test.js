const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');

const ROOT = path.resolve(__dirname, '..');
const script = fs.readFileSync(path.join(ROOT, 'script.js'), 'utf8');
const styles = fs.readFileSync(path.join(ROOT, 'styles.css'), 'utf8');
const postHeaderTemplate = fs.readFileSync(path.join(ROOT, 'partials', 'components', 'post-header.html'), 'utf8');

test('wide display math is measured, scaled proportionally, and recalculated on resize', () => {
    assert.match(script, /mjx-container\[display="true"\]/);
    assert.match(script, /parentWidth\s*\/\s*mathWidth/);
    assert.match(script, /transform\s*=\s*`scale\(\$\{scale\}\)`/);
    assert.match(script, /addEventListener\('resize'/);
    assert.match(styles, /mjx-container\[display="true"\]/);
});

test('inline math avoids justified spacing and oversized formulas get a safe fallback', () => {
    const inlineMathHandler = script.match(/function optimizeInlineMath[\s\S]*?\n}\n\nfunction initResponsiveMath/)?.[0] || '';
    assert.match(script, /function optimizeInlineMath/);
    assert.match(script, /contains-inline-math/);
    assert.match(script, /formulaWidth > availableWidth/);
    assert.doesNotMatch(inlineMathHandler, /math\.scrollWidth/);
    assert.match(script, /inline-math-overflow/);
    assert.match(styles, /\.post-content p\.contains-inline-math[\s\S]*text-align:\s*start/s);
    assert.match(styles, /mjx-container\.inline-math-overflow[\s\S]*overflow-x:\s*auto/s);
});

test('article typography uses the Scientific Spaces font scale consistently', () => {
    assert.match(styles, /--article-font-family:\s*Georgia,[^;]*"Microsoft YaHei"/);
    assert.match(styles, /--article-font-size:\s*14px/);
    assert.match(styles, /--article-line-height:\s*1\.8/);
    assert.match(styles, /--article-aux-font-size:\s*12px/);
    assert.match(styles, /\.single-post-content\s*\{[\s\S]*font-family:\s*var\(--article-font-family\)/s);
    assert.match(styles, /blockquote\s*\{[\s\S]*font-size:\s*inherit/s);
    assert.match(styles, /blockquote p[\s\S]*font-size:\s*inherit/s);
    assert.doesNotMatch(styles, /blockquote(?:\.cite)?[^}]*font-size:\s*(?:13|15)px/s);
    assert.match(styles, /figcaption\s*\{[\s\S]*font-size:\s*var\(--article-aux-font-size\)/s);
});

test('desktop layout keeps a 960px frame and a wider article reading column', () => {
    assert.match(styles, /--layout-max-width:\s*960px/);
    assert.match(styles, /\.sidebar\s*\{[\s\S]*flex:\s*0 0 33\.333%/s);
    assert.match(styles, /\.sidebar\.single-sidebar\s*\{[\s\S]*flex:\s*0 0 30\.5%/s);
    assert.match(styles, /@media \(max-width:\s*768px\)[\s\S]*\.main-content\s*\{[\s\S]*flex-direction:\s*column/s);
});

test('interactive elements retain visible keyboard focus', () => {
    assert.doesNotMatch(styles, /a:focus\s*\{[^}]*outline:\s*none/s);
    assert.match(styles, /a:focus-visible[\s\S]*outline:\s*2px/);
});

test('mobile drawer uses dynamic viewport units and consistent stacking', () => {
    assert.match(styles, /height:\s*100dvh/);
    assert.match(styles, /\.mobile-drawer\s*\{[\s\S]*z-index:\s*1001/s);
    assert.match(styles, /\.mobile-drawer-backdrop\s*\{[\s\S]*z-index:\s*1000/s);
    assert.match(script, /mobileDrawerCloseHandler\s*=\s*close/);
});

test('tables and code use accessible horizontal scroll regions on narrow screens', () => {
    assert.match(script, /function enhanceResponsiveTables/);
    assert.match(script, /wrapper\.tabIndex\s*=\s*0/);
    assert.match(script, /可横向滚动的代码/);
    assert.match(styles, /\.table-scroll\s*\{[\s\S]*overflow-x:\s*auto/s);
    assert.match(styles, /\.code-scroll\s*\{[\s\S]*overflow:\s*auto/s);
    assert.doesNotMatch(styles, /\.post-content table\s*\{[^}]*display:\s*block/s);
});

test('code blocks use one self-contained Scientific Spaces-style theme', () => {
    assert.doesNotMatch(script, /HIGHLIGHT_JS_URL|window\.hljs|data-hljs/);
    assert.match(script, /PRISM_RESOURCES[\s\S]*prismjs@1\.30\.0\/prism\.min\.js/);
    assert.match(script, /integrity:\s*'sha384-/);
    assert.match(script, /classList\.add\('line-numbers'\)/);
    assert.match(script, /function ensureCodeLineNumbers/);
    assert.match(script, /rows\.className\s*=\s*'line-numbers-rows'/);
    assert.match(styles, /--code-bg:\s*#f5f2f0/);
    assert.match(styles, /\.code-block\s*\{[\s\S]*--code-font-size:\s*var\(--article-font-size\)/s);
    assert.match(styles, /\.token\.comment,[\s\S]*color:\s*#708090/s);
    assert.match(styles, /\.token\.keyword\s*\{[\s\S]*color:\s*#07a/s);
    assert.match(styles, /\.line-numbers \.line-numbers-rows[\s\S]*border-right:\s*1px solid #999/s);
});

test('images remain proportional and figures stay within content width', () => {
    assert.match(styles, /\.post-content img\s*\{[\s\S]*max-width:\s*100%[\s\S]*height:\s*auto/s);
    assert.match(styles, /\.post-content figure\s*\{[\s\S]*max-width:\s*100%/s);
    assert.doesNotMatch(styles, /\.post-content img\s*\{[^}]*cursor:\s*pointer/s);
});

test('article pages expose print-to-PDF with an A4 content-only layout', () => {
    assert.match(script, /function initPdfDownload/);
    assert.match(script, /data-download-pdf/);
    assert.match(script, /window\.print\(\)/);
    assert.match(script, /addEventListener\('beforeprint'/);
    assert.match(styles, /@page\s*\{[\s\S]*size:\s*A4 portrait/s);
    assert.match(styles, /@media print\s*\{/);
    assert.match(script, /button\.hidden\s*=\s*false/);
    assert.match(postHeaderTemplate, /data-download-pdf/);
    assert.doesNotMatch(postHeaderTemplate, />下载 PDF<\/button>/);
    assert.match(styles, /\.download-pdf-btn\s*\{[\s\S]*margin-left:\s*auto/s);
    assert.match(styles, /\.download-pdf-btn\s*\{[\s\S]*flex:\s*0 0 22px/s);
    assert.match(styles, /\.download-pdf-btn\s*\{[\s\S]*background-image:\s*url/s);
    assert.match(styles, /\.download-pdf-btn[\s\S]*display:\s*none !important/s);
    assert.match(styles, /\.post-date-badge\s*\{[\s\S]*display:\s*none/s);
    assert.match(styles, /\.post-header\s*\{[\s\S]*background:\s*transparent/s);
    assert.match(styles, /\.code-block code\s*\{[\s\S]*white-space:\s*pre-wrap/s);
});

test('all-posts actions clear every index filter atomically', () => {
    assert.match(script, /function resetIndexQueryParams/);
    assert.match(script, /\['category', 'tag', 'search', 'page'\]/);
    assert.match(script, /if \(isAll\) \{\s*window\.location\.assign\(targetUrl\)/);
    assert.match(script, /if \(!value\) \{\s*window\.location\.assign\(targetUrl\)/);
    assert.match(script, /!activeNormalized && !getTagFilter\(\) && !getSearchQuery\(\)/);
});

test('stale asynchronous index renders cannot overwrite the latest filter state', () => {
    assert.match(script, /let indexRenderVersion = 0/);
    assert.match(script, /const renderVersion = \+\+indexRenderVersion/);
    assert.match(script, /renderVersion === indexRenderVersion/);
    assert.match(script, /if \(!isCurrentRender\(\)\) \{\s*return;/);
});
