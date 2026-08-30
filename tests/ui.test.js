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
