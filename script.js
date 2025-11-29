const componentCache = {};

async function injectPartials(rootPrefix) {
    const includeElements = Array.from(document.querySelectorAll('[data-include]'));
    if (!includeElements.length) {
        return;
    }

    const rootOverride = document.body.getAttribute('data-root');
    const prefix = typeof rootOverride === 'string' && rootOverride.length
        ? rootOverride
        : rootPrefix;

    await Promise.all(includeElements.map(async (element) => {
        const url = element.getAttribute('data-include');
        if (!url) {
            return;
        }

        try {
            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(`Failed to load partial: ${url}`);
            }
            let markup = await response.text();
            markup = markup.replace(/\{\{ROOT\}\}/g, prefix);

            const template = document.createElement('template');
            template.innerHTML = markup.trim();
            const content = template.content.cloneNode(true);
            element.replaceWith(content);
        } catch (err) {
            console.error(err);
        }
    }));
}

async function renderComponents(rootPrefix) {
    const componentElements = Array.from(document.querySelectorAll('[data-component]'));
    if (!componentElements.length) {
        return;
    }

    await Promise.all(componentElements.map(async (element) => {
        const name = element.getAttribute('data-component');
        if (!name) {
            return;
        }

        try {
            const templateString = await loadComponentTemplate(name, rootPrefix);
            const bodyContent = element.innerHTML;
            const data = collectComponentData(element);
            const markup = await buildComponentMarkup(templateString, data, bodyContent, rootPrefix);

            const template = document.createElement('template');
            template.innerHTML = markup.trim();
            const content = template.content.cloneNode(true);
            element.replaceWith(content);
        } catch (err) {
            console.error(err);
        }
    }));
}

async function loadComponentTemplate(name, rootPrefix) {
    if (componentCache[name]) {
        return componentCache[name];
    }

    const response = await fetch(`${rootPrefix}partials/components/${name}.html`);
    if (!response.ok) {
        throw new Error(`Failed to load component: ${name}`);
    }
    const markup = await response.text();
    componentCache[name] = markup;
    return markup;
}

function collectComponentData(element) {
    return Array.from(element.attributes).reduce((acc, attr) => {
        if (attr.name.startsWith('data-') && attr.name !== 'data-component') {
            const key = attr.name
                .replace(/^data-/, '')
                .replace(/-+/g, '_')
                .toUpperCase();
            acc[key] = attr.value;
        }
        return acc;
    }, {});
}

async function buildComponentMarkup(templateString, data, bodyContent, rootPrefix) {
    const titleText = data.TITLE || '';
    const headingTag = (data.HEADING || 'h2').toLowerCase();
    const wrapperClass = data.WRAPPER_CLASS || 'post-content-wrapper';
    const metaClass = data.META_CLASS || 'post-meta';
    const hasTitleClass = Object.prototype.hasOwnProperty.call(data, 'TITLE_CLASS');
    const titleClass = hasTitleClass ? data.TITLE_CLASS : 'post-title';
    const link = data.LINK;
    const resolvedLink = resolveComponentLink(link, rootPrefix);
    const linkOpen = resolvedLink ? `<a href="${resolvedLink}">` : '';
    const linkClose = resolvedLink ? '</a>' : '';
    const headingClassAttr = titleClass ? ` class="${titleClass}"` : '';
    const titleElement = `<${headingTag}${headingClassAttr}>${linkOpen}${titleText}${linkClose}</${headingTag}>`;

    const baseMap = {
        ROOT: rootPrefix,
        BODY: bodyContent || '',
        DAY: data.DAY || '',
        MONTH: data.MONTH || '',
        TITLE: titleText,
        META: data.META || '',
        WRAPPER_CLASS: wrapperClass,
        META_CLASS: metaClass,
        TITLE_ELEMENT: titleElement,
    };

    const combinedMap = { ...data, ...baseMap };

    // First pass: process INCLUDE directives
    let markup = templateString;
    const includePattern = /\{\{INCLUDE:([a-zA-Z0-9_-]+)\}\}/g;
    const includeMatches = [...templateString.matchAll(includePattern)];

    for (const match of includeMatches) {
        const componentName = match[1];
        try {
            const includedTemplate = await loadComponentTemplate(componentName, rootPrefix);
            markup = markup.replace(match[0], includedTemplate);
        } catch (err) {
            console.error(`Failed to include component: ${componentName}`, err);
        }
    }

    // Second pass: replace all other placeholders
    return Object.entries(combinedMap).reduce((result, [key, value]) => {
        const pattern = new RegExp(`\\{\\{${key}\\}\\}`, 'g');
        return result.replace(pattern, () => value ?? '');
    }, markup);
}

function resolveComponentLink(link, rootPrefix) {
    if (!link) {
        return '';
    }
    if (/^(?:[a-z]+:|\/\/|#)/i.test(link)) {
        return link;
    }
    return `${rootPrefix}${link}`;
}

function fixPostMetaLinks(rootPrefix) {
    // Find the post metadata JSON
    const metadataScript = document.getElementById('post-metadata');
    if (!metadataScript) {
        return; // Not a post page
    }

    let metadata;
    try {
        metadata = JSON.parse(metadataScript.textContent);
    } catch (error) {
        console.error('Failed to parse post metadata:', error);
        return;
    }

    // Find the post-meta div (should be inside post-header after rendering)
    const postMeta = document.querySelector('.post-header .post-meta, .post-meta');
    if (!postMeta) {
        return;
    }

    // Fix category links
    const categoryLinks = postMeta.querySelectorAll('.meta-categories a[href="#"]');
    const categories = Array.isArray(metadata.categories) ? metadata.categories : [];
    categoryLinks.forEach((link, index) => {
        if (index < categories.length) {
            link.href = buildCategoryHref(rootPrefix, categories[index]);
        }
    });

    // Fix tag links
    const tagLinks = postMeta.querySelectorAll('.meta-tags a[href="#"]');
    const tags = Array.isArray(metadata.tags) ? metadata.tags : [];
    tagLinks.forEach((link, index) => {
        if (index < tags.length) {
            link.href = buildTagHref(rootPrefix, tags[index]);
        }
    });
}

const POSTS_PER_PAGE = 10;
const SUMMARY_ALLOWED_TAGS = new Set([
    'p',
    'h2',
    'h3',
    'h4',
    'h5',
    'ul',
    'ol',
    'li',
    'blockquote',
    'table',
    'figure',
]);
let mathJaxReadyPromise;
const HIGHLIGHT_CSS_URL = 'https://cdn.jsdelivr.net/gh/highlightjs/cdn-release/build/styles/github.css';
const HIGHLIGHT_JS_URL = 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js';
let highlightReadyPromise;
let highlightConfigured = false;
let manifestFetchPromise = null;
let cachedPostList = null;

async function initIndexPage(rootPrefix) {
    const listContainer = document.querySelector('[data-post-list]');
    if (!listContainer) {
        return;
    }

    const paginationContainer = document.querySelector('[data-pagination]');
    setLoadingState(listContainer);

    try {
        const posts = await loadAndCachePosts(rootPrefix);
        const searchQuery = getSearchQuery();
        const categoryFilter = getCategoryFilter();
        const tagFilter = getTagFilter();

        await initCategoryNav(rootPrefix, posts, categoryFilter);
        await initTagCloud(rootPrefix, posts, tagFilter);
        // Index page shows random posts in sidebar
        await initRandomPosts(rootPrefix, posts);

        if (searchQuery) {
            await renderSearchResults(listContainer, paginationContainer, posts, searchQuery, rootPrefix);
            await initTagCloud(rootPrefix, posts, tagFilter);
            await typesetMath(listContainer);
            return;
        }

        let filteredPosts = posts.slice();
        if (categoryFilter) {
            filteredPosts = filterPostsByCategory(filteredPosts, categoryFilter);
        }
        if (tagFilter) {
            filteredPosts = filterPostsByTag(filteredPosts, tagFilter);
        }

        if (!filteredPosts.length) {
            if (categoryFilter && tagFilter) {
                const categoryName = resolveCategoryDisplayName(posts, categoryFilter);
                const tagName = resolveTagDisplayName(posts, tagFilter);
                listContainer.innerHTML = `<p class="post-list-placeholder">分类 “${escapeHtml(categoryName)}” 与标签 “${escapeHtml(tagName)}” 下暂无文章。</p>`;
            } else if (categoryFilter) {
                const displayName = resolveCategoryDisplayName(posts, categoryFilter);
                listContainer.innerHTML = `<p class="post-list-placeholder">分类 “${escapeHtml(displayName)}” 下暂无文章。</p>`;
            } else if (tagFilter) {
                const tagName = resolveTagDisplayName(posts, tagFilter);
                listContainer.innerHTML = `<p class="post-list-placeholder">标签 “${escapeHtml(tagName)}” 下暂无文章。</p>`;
            } else {
                listContainer.innerHTML = '<p class="post-list-placeholder">暂无文章。</p>';
            }
            if (paginationContainer) {
                paginationContainer.innerHTML = '';
                paginationContainer.setAttribute('hidden', '');
            }
            return;
        }

        const totalPages = Math.max(1, Math.ceil(filteredPosts.length / POSTS_PER_PAGE));
        const requestedPage = getRequestedPage();
        const currentPage = Math.min(Math.max(requestedPage, 1), totalPages);
        const startIndex = (currentPage - 1) * POSTS_PER_PAGE;
        const pagePosts = filteredPosts.slice(startIndex, startIndex + POSTS_PER_PAGE);

        listContainer.innerHTML = '';

        pagePosts.forEach((post) => {
            listContainer.appendChild(createPostArticle(post, rootPrefix));
        });

        renderPagination(paginationContainer, currentPage, totalPages);
        await renderComponents(rootPrefix);
        enhanceCodeBlocks(listContainer);
        await highlightCodeBlocks(listContainer);
        await typesetMath(listContainer);
    } catch (error) {
        console.error('Failed to load posts manifest', error);
        listContainer.innerHTML = '<p class="post-list-placeholder">文章加载失败，请稍后重试。</p>';
        if (paginationContainer) {
            paginationContainer.innerHTML = '';
            paginationContainer.setAttribute('hidden', '');
        }
    }
}

function setLoadingState(container) {
    container.innerHTML = '<p class="post-list-placeholder">正在加载文章...</p>';
}

async function fetchPostManifest(rootPrefix) {
    const manifestUrl = `${rootPrefix}posts/manifest.json`;
    const response = await fetch(manifestUrl, { cache: 'no-store' });
    if (!response.ok) {
        throw new Error(`Failed to load manifest: ${manifestUrl}`);
    }
    return response.json();
}

async function loadAndCachePosts(rootPrefix) {
    if (Array.isArray(cachedPostList)) {
        return cachedPostList;
    }

    if (!manifestFetchPromise) {
        manifestFetchPromise = fetchPostManifest(rootPrefix)
            .then((manifest) => normalizeManifestPosts(manifest))
            .then((posts) => {
                cachedPostList = Array.isArray(posts) ? posts : [];
                return cachedPostList;
            })
            .catch((error) => {
                manifestFetchPromise = null;
                throw error;
            });
    }

    const posts = await manifestFetchPromise;
    if (!Array.isArray(cachedPostList)) {
        cachedPostList = Array.isArray(posts) ? posts : [];
    }
    return cachedPostList;
}

function normalizeManifestPosts(manifest) {
    if (!manifest || !Array.isArray(manifest.posts)) {
        return [];
    }

    return manifest.posts
        .slice()
        .sort((a, b) => {
            const aTime = a && a.createdAt ? new Date(a.createdAt).getTime() : 0;
            const bTime = b && b.createdAt ? new Date(b.createdAt).getTime() : 0;
            return bTime - aTime;
        });
}

function getRequestedPage() {
    const params = new URLSearchParams(window.location.search);
    const value = Number.parseInt(params.get('page'), 10);
    return Number.isNaN(value) ? 1 : value;
}

function createPostArticle(post, rootPrefix) {
    const article = document.createElement('article');
    article.className = 'post';

    const card = document.createElement('div');
    card.setAttribute('data-component', 'post-card');
    card.setAttribute('data-day', post.day || '');
    card.setAttribute('data-month', post.month || '');
    card.setAttribute('data-title', post.title || '');
    card.setAttribute('data-link', post.link || '');
    card.setAttribute('data-meta', buildPostMetaHtml(post, rootPrefix));

    const summaryHtml = buildSummaryHtml(post.summary, rootPrefix);
    const readMoreHref = resolveComponentLink(post.link, rootPrefix) || '#';

    card.innerHTML = `
        <div class="post-content">
            ${summaryHtml}
            <p><a href="${readMoreHref}" class="read-more">点击阅读全文...</a></p>
        </div>
    `;

    article.appendChild(card);
    return article;
}

function buildPostMetaHtml(post, rootPrefix) {
    const categories = Array.isArray(post && post.categories) ? post.categories : [];
    const categoryHtml = categories.length
        ? categories
            .map((category) => {
                const href = buildCategoryHref(rootPrefix, category);
                return `<a href="${href}">${escapeHtml(category)}</a>`;
            })
            .join(',')
        : '<span class="post-taxonomy-empty">未分类</span>';
    const catSpan = `<span class="meta-item meta-categories">分类：${categoryHtml}</span>`;

    const tags = Array.isArray(post && post.tags) ? post.tags : [];
    const tagHtml = tags.length
        ? tags
            .map((tag) => {
                const href = buildTagHref(rootPrefix, tag);
                return `<a href="${href}">${escapeHtml(tag)}</a>`;
            })
            .join(',')
        : '<span class="post-taxonomy-empty">暂无标签</span>';
    const tagSpan = `<span class="meta-item meta-tags">标签：${tagHtml}</span>`;

    if (categories.length && tags.length) {
        return `${catSpan} <span class="meta-divider">|</span> ${tagSpan}`;
    }
    return `${catSpan} ${tagSpan}`;
}

function buildSummaryHtml(summary, rootPrefix) {
    if (!Array.isArray(summary) || !summary.length) {
        return '<p>暂无摘要。</p>';
    }

    const combined = summary
        .map((item) => {
            if (!item || typeof item.html !== 'string') {
                return '';
            }
            const desiredTag = typeof item.type === 'string' ? item.type.toLowerCase() : 'p';
            const tag = SUMMARY_ALLOWED_TAGS.has(desiredTag) ? desiredTag : 'p';
            const normalizedHtml = normalizeLatexEscapes(item.html);
            const trimmed = normalizedHtml.trim();
            const openPattern = new RegExp(`^<${tag}(?=\s|>)`, 'i');
            const closePattern = new RegExp(`</${tag}>\s*$`, 'i');

            if (openPattern.test(trimmed) && closePattern.test(trimmed)) {
                return trimmed;
            }

            return `<${tag}>${normalizedHtml}</${tag}>`;
        })
        .join('\n');

    return rewriteSummaryAssetPaths(combined, rootPrefix);
}

function normalizeLatexEscapes(html) {
    if (typeof html !== 'string') {
        return html;
    }
    return html.replace(/\\\\(?=\S)/g, '\\');
}

function rewriteSummaryAssetPaths(html, rootPrefix) {
    if (typeof html !== 'string' || !html.length) {
        return html;
    }
    if (rootPrefix === '') {
        return html
            .replace(/(\bsrc\s*=\s*["'])\.\.\/assets\//gi, '$1assets/')
            .replace(/(\bsrcset\s*=\s*["'][^"']*)\.\.\/assets\//gi, (m) => m.replace(/\.\.\/assets\//gi, 'assets/'));
    }
    return html;
}

function enhanceCodeBlocks(root) {
    if (typeof document === 'undefined') {
        return;
    }

    const container = root && typeof root.querySelectorAll === 'function' ? root : document;
    const preElements = container.querySelectorAll('pre');

    preElements.forEach((pre) => {
        if (pre.closest('.code-block')) {
            return;
        }

        if (pre.dataset.codeEnhanced === 'true') {
            return;
        }

        const parent = pre.parentNode;
        if (!parent) {
            return;
        }

        const wrapper = document.createElement('div');
        wrapper.className = 'code-block';

        const copyButton = document.createElement('button');
        copyButton.type = 'button';
        copyButton.className = 'code-copy-btn';
        copyButton.dataset.originalLabel = '复制代码';
        copyButton.setAttribute('aria-label', '复制代码');
        copyButton.title = '复制代码';

        const codeBody = document.createElement('div');
        codeBody.className = 'code-body';

        const codeScroll = document.createElement('div');
        codeScroll.className = 'code-scroll';

        parent.insertBefore(wrapper, pre);
        pre.dataset.codeEnhanced = 'true';

        codeScroll.appendChild(pre);
        codeBody.appendChild(codeScroll);
        wrapper.appendChild(copyButton);
        wrapper.appendChild(codeBody);
    });
}

async function ensureHighlightResources() {
    if (typeof window === 'undefined' || typeof document === 'undefined') {
        return null;
    }

    if (window.hljs && typeof window.hljs.highlightElement === 'function') {
        return window.hljs;
    }

    if (highlightReadyPromise) {
        return highlightReadyPromise;
    }

    highlightReadyPromise = new Promise((resolve, reject) => {
        const resolveIfReady = () => {
            if (window.hljs && typeof window.hljs.highlightElement === 'function') {
                resolve(window.hljs);
            } else {
                reject(new Error('Highlight.js failed to initialize'));
            }
        };

        if (!document.querySelector('link[data-hljs="theme"]')) {
            const link = document.createElement('link');
            link.rel = 'stylesheet';
            link.href = HIGHLIGHT_CSS_URL;
            link.dataset.hljs = 'theme';
            document.head.appendChild(link);
        }

        if (document.querySelector('script[data-hljs="library"]')) {
            const existing = document.querySelector('script[data-hljs="library"]');
            existing.addEventListener('load', resolveIfReady, { once: true });
            existing.addEventListener('error', reject, { once: true });
            return;
        }

        const script = document.createElement('script');
        script.src = HIGHLIGHT_JS_URL;
        script.async = true;
        script.defer = true;
        script.dataset.hljs = 'library';
        script.addEventListener('load', () => {
            window.setTimeout(resolveIfReady, 0);
        }, { once: true });
        script.addEventListener('error', reject, { once: true });
        document.head.appendChild(script);
    })
        .catch((error) => {
            console.error('Highlight.js load failed', error);
            highlightReadyPromise = null;
            return null;
        });

    return highlightReadyPromise;
}

async function highlightCodeBlocks(root) {
    const hljs = await ensureHighlightResources();
    if (!hljs) {
        return;
    }

    if (!highlightConfigured) {
        hljs.configure({ ignoreUnescapedHTML: true });
        highlightConfigured = true;
    }

    const container = root && typeof root.querySelectorAll === 'function' ? root : document;
    const codeElements = container.querySelectorAll('pre code, code.language-plaintext');

    codeElements.forEach((codeEl) => {
        if (codeEl.dataset.highlighted === 'true') {
            return;
        }

        const classList = Array.from(codeEl.classList || []);
        const hasLanguageClass = classList.some((cls) => cls.startsWith('language-') || cls.startsWith('lang-'));
        if (!hasLanguageClass) {
            codeEl.classList.add('language-plaintext');
        }

        hljs.highlightElement(codeEl);
        codeEl.dataset.highlighted = 'true';
    });
}

async function handleCodeCopyClick(event) {
    const button = event.target.closest('.code-copy-btn');
    if (!button) {
        return;
    }

    const block = button.closest('.code-block');
    if (!block) {
        return;
    }

    const pre = block.querySelector('pre');
    if (!pre) {
        return;
    }

    const codeText = pre.innerText;
    const originalLabel = button.dataset.originalLabel || '复制代码';
    const setButtonLabel = (value) => {
        button.setAttribute('aria-label', value);
        button.title = value;
    };

    button.disabled = true;
    button.classList.remove('copied', 'copy-error');
    setButtonLabel('复制中…');

    try {
        await writeClipboard(codeText);
        button.classList.add('copied');
        setButtonLabel('复制成功');
    } catch (error) {
        console.error('复制代码失败', error);
        button.classList.add('copy-error');
        setButtonLabel('复制失败');
    }

    window.setTimeout(() => {
        button.classList.remove('copied', 'copy-error');
        setButtonLabel(originalLabel);
        button.disabled = false;
    }, 2000);
}

function initFooterYear() {
    const el = document.querySelector('.footer [data-year-range]');
    if (!el) {
        return;
    }
    const startYear = 2009;
    const currentYear = new Date().getFullYear();
    el.textContent = currentYear > startYear ? `${startYear}-${currentYear}` : `${startYear}`;
}

async function writeClipboard(text) {
    if (typeof navigator !== 'undefined' && navigator.clipboard && typeof navigator.clipboard.writeText === 'function') {
        await navigator.clipboard.writeText(text);
        return;
    }

    const textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.style.position = 'fixed';
    textarea.style.opacity = '0';
    textarea.style.pointerEvents = 'none';
    document.body.appendChild(textarea);
    textarea.focus();
    textarea.select();

    const successful = document.execCommand('copy');
    document.body.removeChild(textarea);
    if (!successful) {
        throw new Error('document.execCommand("copy") failed');
    }
}

function escapeHtml(value) {
    const text = String(value ?? '');
    return text.replace(/[&<>"']/g, (char) => {
        switch (char) {
            case '&':
                return '&amp;';
            case '<':
                return '&lt;';
            case '>':
                return '&gt;';
            case '"':
                return '&quot;';
            case '\'':
                return '&#39;';
            default:
                return char;
        }
    });
}

function stripHtmlTags(value) {
    if (!value) {
        return '';
    }
    return String(value).replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim();
}

function buildSummaryText(post) {
    if (!post) {
        return '';
    }
    if (Array.isArray(post.summary)) {
        return post.summary
            .map((item) => (item && typeof item.html === 'string' ? stripHtmlTags(item.html) : ''))
            .filter(Boolean)
            .join(' ');
    }
    if (typeof post.summary === 'string') {
        return stripHtmlTags(post.summary);
    }
    return '';
}

function truncateText(value, maxLength) {
    const text = typeof value === 'string' ? value.trim() : '';
    if (!text) {
        return '';
    }
    if (text.length <= maxLength) {
        return text;
    }
    return `${text.slice(0, maxLength).replace(/\s+$/u, '')}…`;
}

function normalizeCategoryValue(value) {
    return String(value ?? '').trim().toLowerCase();
}

function normalizeTagValue(value) {
    return String(value ?? '').trim().toLowerCase();
}

const TAG_ALIASES = {};

function canonicalizeTagLabel(value) {
    const raw = String(value ?? '').trim();
    const key = normalizeTagValue(raw);
    const mapped = TAG_ALIASES[key];
    if (mapped) {
        return mapped;
    }
    if (raw === '计算成像' || raw === '人工智能' || raw === '工程实践' || raw === '数学研究') {
        return raw;
    }
    return raw;
}

function normalizeCanonicalTagValue(value) {
    return normalizeTagValue(canonicalizeTagLabel(value));
}

const CATEGORY_CONFIG = [
    { key: '计算成像', label: '计算成像' },
    { key: '人工智能', label: '人工智能' },
    { key: '工程实践', label: '工程实践' },
    { key: '数学研究', label: '数学研究' },
];

const CATEGORY_DISPLAY_OVERRIDES = CATEGORY_CONFIG.reduce((acc, item) => {
    const key = normalizeCategoryValue(item.key);
    acc[key] = { label: item.label };
    return acc;
}, {});

const CATEGORY_DISPLAY_ORDER = CATEGORY_CONFIG.map((item) => normalizeCategoryValue(item.key));

function resolveCategoryDisplayName(posts, rawCategory) {
    const target = normalizeCategoryValue(rawCategory);
    if (!target) {
        return '';
    }

    for (const post of posts || []) {
        const categories = Array.isArray(post?.categories) ? post.categories : [];
        const match = categories.find((category) => normalizeCategoryValue(category) === target);
        if (match) {
            return match;
        }
    }

    return rawCategory || '';
}

const CATEGORY_ICON_SRC_MAP = {
    '计算成像': 'ct-scan.png',
    '人工智能': 'technology.png',
    '工程实践': 'implementation.png',
    '数学研究': 'algorithm.png',
    all: 'all.png',
};

function getCategoryIconSrc(name, rootPrefix, isAll = false) {
    const key = isAll ? 'all' : normalizeCategoryValue(name);
    const file = CATEGORY_ICON_SRC_MAP[key] || 'math.png';
    return `${rootPrefix}icons/${file}`;
}

function filterPostsByCategory(posts, category) {
    const target = normalizeCategoryValue(category);
    if (!target) {
        return Array.isArray(posts) ? posts.slice() : [];
    }

    return (posts || []).filter((post) => {
        const categories = Array.isArray(post?.categories) ? post.categories : [];
        return categories.some((item) => normalizeCategoryValue(item) === target);
    });
}

function resolveTagDisplayName(posts, rawTag) {
    const label = canonicalizeTagLabel(rawTag);
    return label || '';
}

function filterPostsByTag(posts, tag) {
    const target = normalizeCanonicalTagValue(tag);
    if (!target) {
        return Array.isArray(posts) ? posts.slice() : [];
    }

    return (posts || []).filter((post) => {
        const tags = Array.isArray(post?.tags) ? post.tags : [];
        return tags.some((item) => normalizeCanonicalTagValue(item) === target);
    });
}

function getCategoryDisplayMeta(categoryName, _count, isAll = false) {
    if (isAll) {
        return {
            label: '全部文章',
        };
    }

    const normalized = normalizeCategoryValue(categoryName);
    const override = CATEGORY_DISPLAY_OVERRIDES[normalized];
    const label = override?.label || categoryName;

    return {
        label,
    };
}

function searchPosts(posts, query) {
    if (!Array.isArray(posts) || !posts.length || !query) {
        return [];
    }

    const keywords = query
        .trim()
        .toLowerCase()
        .split(/\s+/)
        .filter(Boolean);

    if (!keywords.length) {
        return [];
    }

    const results = [];

    posts.forEach((post) => {
        if (!post) {
            return;
        }
        const title = (post.title || '').toLowerCase();
        const categories = Array.isArray(post.categories) ? post.categories.join(' ').toLowerCase() : '';
        const tags = Array.isArray(post.tags) ? post.tags.join(' ').toLowerCase() : '';
        const summary = buildSummaryText(post).toLowerCase();

        let score = 0;
        keywords.forEach((keyword) => {
            if (!keyword) {
                return;
            }
            if (title.includes(keyword)) {
                score += 5;
            }
            if (categories.includes(keyword)) {
                score += 2;
            }
            if (tags.includes(keyword)) {
                score += 2;
            }
            if (summary.includes(keyword)) {
                score += 1;
            }
        });

        if (score > 0) {
            results.push({ post, score });
        }
    });

    results.sort((a, b) => {
        if (b.score !== a.score) {
            return b.score - a.score;
        }
        const aTime = a.post && a.post.createdAt ? new Date(a.post.createdAt).getTime() : 0;
        const bTime = b.post && b.post.createdAt ? new Date(b.post.createdAt).getTime() : 0;
        return bTime - aTime;
    });

    return results;
}

function setSearchStatus(element, message) {
    if (!element) {
        return;
    }
    if (!message) {
        element.textContent = '';
        element.setAttribute('hidden', '');
    } else {
        element.textContent = message;
        element.removeAttribute('hidden');
    }
}

function getSearchQuery() {
    if (typeof window === 'undefined') {
        return '';
    }
    const params = new URLSearchParams(window.location.search);
    return (params.get('search') || '').trim();
}

function getCategoryFilter() {
    if (typeof window === 'undefined') {
        return '';
    }
    const params = new URLSearchParams(window.location.search);
    const value = params.get('category');
    return value ? value.trim() : '';
}

function getTagFilter() {
    if (typeof window === 'undefined') {
        return '';
    }
    const params = new URLSearchParams(window.location.search);
    const value = params.get('tag');
    return value ? canonicalizeTagLabel(value.trim()) : '';
}

function updateSearchQueryParam(query) {
    if (typeof window === 'undefined') {
        return;
    }
    const url = new URL(window.location.href);
    if (!query) {
        url.searchParams.delete('search');
    } else {
        url.searchParams.set('search', query);
        url.searchParams.delete('category');
        url.searchParams.delete('tag');
    }
    url.searchParams.delete('page');
    window.history.replaceState({}, '', `${url.pathname}${url.search}${url.hash}`);
}

function updateCategoryQueryParam(category) {
    if (typeof window === 'undefined') {
        return;
    }
    const url = new URL(window.location.href);
    if (!category) {
        url.searchParams.delete('category');
    } else {
        url.searchParams.set('category', category);
        url.searchParams.delete('search');
    }
    url.searchParams.delete('page');
    window.history.replaceState({}, '', `${url.pathname}${url.search}${url.hash}`);
}

function updateTagQueryParam(tag) {
    if (typeof window === 'undefined') {
        return;
    }
    const url = new URL(window.location.href);
    if (!tag) {
        url.searchParams.delete('tag');
    } else {
        url.searchParams.set('tag', canonicalizeTagLabel(tag));
        url.searchParams.delete('search');
    }
    url.searchParams.delete('page');
    window.history.replaceState({}, '', `${url.pathname}${url.search}${url.hash}`);
}

function buildIndexSearchUrl(rootPrefix, query) {
    const base = rootPrefix ? `${rootPrefix}index.html` : 'index.html';
    return `${base}?search=${encodeURIComponent(query)}`;
}

function buildCategoryHref(rootPrefix, category) {
    const base = rootPrefix ? `${rootPrefix}index.html` : 'index.html';
    if (!category) {
        return base;
    }
    return `${base}?category=${encodeURIComponent(category)}`;
}

function buildTagHref(rootPrefix, tag) {
    const base = rootPrefix ? `${rootPrefix}index.html` : 'index.html';
    if (!tag) {
        return base;
    }
    const label = canonicalizeTagLabel(tag);
    return `${base}?tag=${encodeURIComponent(label)}`;
}

async function renderSearchResults(listContainer, paginationContainer, posts, query, rootPrefix) {
    if (!listContainer) {
        return [];
    }

    const matches = searchPosts(posts, query);
    listContainer.innerHTML = '';

    if (paginationContainer) {
        paginationContainer.innerHTML = '';
        paginationContainer.setAttribute('hidden', '');
    }

    if (!matches.length) {
        listContainer.innerHTML = `<p class="post-list-placeholder">未找到与“${escapeHtml(query)}”相关的文章。</p>`;
        return matches;
    }

    const heading = document.createElement('p');
    heading.className = 'search-results-heading';
    heading.innerHTML = `找到 <strong>${matches.length}</strong> 篇与 “${escapeHtml(query)}” 相关的文章：`;
    listContainer.appendChild(heading);

    matches.forEach(({ post }) => {
        listContainer.appendChild(createPostArticle(post, rootPrefix));
    });

    await renderComponents(rootPrefix);
    enhanceCodeBlocks(listContainer);
    await highlightCodeBlocks(listContainer);
    await initTagCloud(rootPrefix, posts, getTagFilter());

    return matches;
}

async function initCategoryNav(rootPrefix, postsArg, activeCategory) {
    const nav = document.querySelector('[data-category-nav]');
    if (!nav) {
        return;
    }

    let posts = postsArg;
    if (!Array.isArray(posts)) {
        try {
            posts = await loadAndCachePosts(rootPrefix);
        } catch (error) {
            console.error('加载分类数据失败', error);
            nav.innerHTML = '';
            return;
        }
    }

    const totalCount = Array.isArray(posts) ? posts.length : 0;
    const counts = new Map();

    (posts || []).forEach((post) => {
        const categories = Array.isArray(post?.categories) ? post.categories : [];
        categories.forEach((category) => {
            const displayName = String(category || '').trim();
            if (!displayName) {
                return;
            }
            const key = normalizeCategoryValue(displayName);
            if (!counts.has(key)) {
                counts.set(key, { name: displayName, count: 0 });
            }
            counts.get(key).count += 1;
        });
    });

    let activeNormalized = normalizeCategoryValue(activeCategory);

    const priorityMap = CATEGORY_DISPLAY_ORDER.reduce((acc, key, index) => {
        acc[key] = index;
        return acc;
    }, {});

    const entries = Array.from(counts.values()).sort((a, b) => {
        const keyA = normalizeCategoryValue(a.name);
        const keyB = normalizeCategoryValue(b.name);
        const hasPriorityA = Object.prototype.hasOwnProperty.call(priorityMap, keyA);
        const hasPriorityB = Object.prototype.hasOwnProperty.call(priorityMap, keyB);

        if (hasPriorityA && hasPriorityB) {
            return priorityMap[keyA] - priorityMap[keyB];
        }
        if (hasPriorityA) {
            return -1;
        }
        if (hasPriorityB) {
            return 1;
        }
        if (b.count !== a.count) {
            return b.count - a.count;
        }
        return a.name.localeCompare(b.name, 'zh-Hans');
    });

    if (activeNormalized) {
        const availableKeys = new Set(entries.map((entry) => normalizeCategoryValue(entry.name)));
        if (!availableKeys.has(activeNormalized)) {
            activeNormalized = '';
        }
    }

    nav.innerHTML = '';

    const appendEntry = (value, count, isActive, isAll = false) => {
        const link = document.createElement('a');
        link.className = 'cat-item';
        link.href = buildCategoryHref(rootPrefix, value);

        const displayMeta = getCategoryDisplayMeta(isAll ? '全部文章' : value, count, isAll);
        const iconSrc = getCategoryIconSrc(isAll ? 'all' : value, rootPrefix, isAll);
        const safeLabel = escapeHtml(displayMeta.label);
        const fallbackSrc = `${rootPrefix}icons/math.png`;
        link.innerHTML = `<img class="icon-img" src="${iconSrc}" alt="${safeLabel}" onerror="this.onerror=null;this.src='${fallbackSrc}'"><span class="label">${safeLabel}</span>`;

        if (isAll) {
            link.dataset.category = 'all';
            link.title = displayMeta.label;
        } else {
            const normalizedValue = normalizeCategoryValue(value);
            link.dataset.category = normalizedValue || 'uncategorized';
            link.title = displayMeta.label;
        }

        if (isActive) {
            link.classList.add('active');
        }

        link.addEventListener('click', async (event) => {
            event.preventDefault();

            const listContainer = document.querySelector('[data-post-list]');
            const targetUrl = buildCategoryHref(rootPrefix, value);

            if (listContainer) {
                updateCategoryQueryParam(value);
                updateSearchQueryParam('');
                await initIndexPage(rootPrefix);
                window.scrollTo({ top: 0, behavior: 'smooth' });
            } else {
                window.location.href = targetUrl;
            }
        });

        nav.appendChild(link);
    };

    appendEntry('', totalCount, !activeNormalized, true);

    entries.forEach(({ name, count }) => {
        const normalizedName = normalizeCategoryValue(name);
        const isActive = Boolean(activeNormalized) && normalizedName === activeNormalized;
        appendEntry(name, count, isActive, false);
    });
}

async function initTagCloud(rootPrefix, postsArg, activeTag) {
    const sections = Array.from(document.querySelectorAll('[data-tag-section]'));
    if (!sections.length) {
        return;
    }

    let posts = postsArg;
    if (!Array.isArray(posts)) {
        try {
            posts = await loadAndCachePosts(rootPrefix);
        } catch (error) {
            console.error('加载标签数据失败', error);
            sections.forEach((section) => {
                section.setAttribute('hidden', '');
                const cloud = section.querySelector('[data-tag-cloud]');
                if (cloud) {
                    cloud.innerHTML = '';
                }
            });
            return;
        }
    }

    const tagMap = new Map();

    (posts || []).forEach((post) => {
        const tags = Array.isArray(post?.tags) ? post.tags : [];
        tags.forEach((rawTag) => {
            const displayName = String(rawTag || '').trim();
            if (!displayName) {
                return;
            }
            const label = canonicalizeTagLabel(displayName);
            const key = normalizeTagValue(label);
            if (!tagMap.has(key)) {
                tagMap.set(key, { name: label, count: 0 });
            }
            tagMap.get(key).count += 1;
        });
    });

    let activeNormalized = normalizeCanonicalTagValue(activeTag);
    if (activeNormalized && !tagMap.has(activeNormalized)) {
        activeNormalized = '';
    }

    const entries = Array.from(tagMap.values()).sort((a, b) => {
        if (b.count !== a.count) {
            return b.count - a.count;
        }
        return a.name.localeCompare(b.name, 'zh-Hans');
    });

    sections.forEach((section) => {
        const cloud = section.querySelector('[data-tag-cloud]');
        if (!cloud) {
            return;
        }

        cloud.innerHTML = '';

        if (!entries.length) {
            section.setAttribute('hidden', '');
            return;
        }

        section.removeAttribute('hidden');

        const appendTag = (label, value, isActive) => {
            const link = document.createElement('a');
            link.className = 'tag-item';
            link.href = buildTagHref(rootPrefix, value);
            link.textContent = label;

            if (isActive) {
                link.classList.add('active');
            }

            link.addEventListener('click', async (event) => {
                event.preventDefault();

                const listContainer = document.querySelector('[data-post-list]');
                const targetUrl = buildTagHref(rootPrefix, value);

                if (listContainer) {
                    updateSearchQueryParam('');
                    updateTagQueryParam(value);
                    await initIndexPage(rootPrefix);
                    window.scrollTo({ top: 0, behavior: 'smooth' });
                } else {
                    window.location.href = targetUrl;
                }
            });

            cloud.appendChild(link);
        };

        appendTag('全部标签', '', !activeNormalized);

        entries.forEach(({ name }) => {
            const normalized = normalizeTagValue(name);
            const isActive = Boolean(activeNormalized) && normalized === activeNormalized;
            appendTag(name, name, isActive);
        });
    });
}

async function initRecentPosts(rootPrefix, postsArg) {
    const containers = Array.from(document.querySelectorAll('[data-recent-posts]'));
    if (!containers.length) {
        return;
    }

    let posts = postsArg;
    if (!Array.isArray(posts)) {
        try {
            posts = await loadAndCachePosts(rootPrefix);
        } catch (error) {
            console.error('加载最新文章失败', error);
            containers.forEach((c) => { c.innerHTML = '<li><a href="#">加载失败</a></li>'; });
            return;
        }
    }

    if (!posts || !posts.length) {
        containers.forEach((c) => { c.innerHTML = '<li><a href="#">暂无文章</a></li>'; });
        return;
    }

    const sortedPosts = posts
        .slice()
        .sort((a, b) => {
            const aTime = a.createdAt ? new Date(a.createdAt).getTime() : 0;
            const bTime = b.createdAt ? new Date(b.createdAt).getTime() : 0;
            return bTime - aTime;
        })
        .slice(0, 5);

    containers.forEach((container) => {
        container.innerHTML = '';
        sortedPosts.forEach((post) => {
            const li = document.createElement('li');
            const link = document.createElement('a');
            link.href = resolveComponentLink(post.link, rootPrefix);
            link.textContent = post.title || '未命名文章';
            li.appendChild(link);
            container.appendChild(li);
        });
    });
}

async function initRandomPosts(rootPrefix, postsArg) {
    const container = document.querySelector('[data-random-posts]');
    if (!container) {
        return;
    }

    let posts = postsArg;
    if (!Array.isArray(posts)) {
        try {
            posts = await loadAndCachePosts(rootPrefix);
        } catch (error) {
            console.error('加载随机文章失败', error);
            container.innerHTML = '<li><a href="#">加载失败</a></li>';
            return;
        }
    }

    if (!posts || !posts.length) {
        container.innerHTML = '<li><a href="#">暂无文章</a></li>';
        return;
    }

    // Shuffle posts and get 5 random ones
    const shuffled = posts.slice();
    for (let i = shuffled.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    const randomPosts = shuffled.slice(0, 5);

    container.innerHTML = '';
    randomPosts.forEach((post) => {
        const li = document.createElement('li');
        const link = document.createElement('a');
        link.href = resolveComponentLink(post.link, rootPrefix);
        link.textContent = post.title || '未命名文章';
        li.appendChild(link);
        container.appendChild(li);
    });
}

async function initPostNavigation(rootPrefix) {
    const navComponents = document.querySelectorAll('[data-component="post-navigation"]');
    if (!navComponents.length) {
        return;
    }

    // Get current post metadata
    const metadataScript = document.getElementById('post-metadata');
    if (!metadataScript) {
        return;
    }

    let currentPost;
    try {
        currentPost = JSON.parse(metadataScript.textContent);
    } catch (error) {
        console.error('解析当前文章元数据失败', error);
        return;
    }

    // Load all posts
    let allPosts;
    try {
        allPosts = await loadAndCachePosts(rootPrefix);
    } catch (error) {
        console.error('加载文章列表失败', error);
        return;
    }

    // Sort posts by date (newest first)
    const sortedPosts = allPosts.slice().sort((a, b) => {
        return new Date(b.createdAt) - new Date(a.createdAt);
    });

    // Find current post index
    const currentIndex = sortedPosts.findIndex(post => post.id === currentPost.id);
    if (currentIndex === -1) {
        console.error('当前文章未在列表中找到');
        return;
    }

    // Get previous and next posts
    // 上一篇 = 更新的文章 (index - 1)
    // 下一篇 = 更旧的文章 (index + 1)
    const prevPost = currentIndex > 0 ? sortedPosts[currentIndex - 1] : null;
    const nextPost = currentIndex < sortedPosts.length - 1 ? sortedPosts[currentIndex + 1] : null;

    // Generate HTML for navigation links
    const prevLink = prevPost
        ? `<span class="prev-post">上一篇：<a href="${resolveComponentLink(prevPost.link, rootPrefix)}">${prevPost.title}</a></span>`
        : '';

    const nextLink = nextPost
        ? `<span class="next-post">下一篇：<a href="${resolveComponentLink(nextPost.link, rootPrefix)}">${nextPost.title}</a></span>`
        : '';

    // Update each navigation component
    navComponents.forEach(component => {
        component.setAttribute('data-prev-post-link', prevLink);
        component.setAttribute('data-next-post-link', nextLink);
    });
}

function initSidebarSearch(rootPrefix) {
    const sections = Array.from(document.querySelectorAll('[data-search-section]'));
    if (!sections.length) {
        return;
    }

    const listContainer = document.querySelector('[data-post-list]');
    const paginationContainer = document.querySelector('[data-pagination]');
    const onIndexPage = Boolean(listContainer);
    const initialQuery = getSearchQuery();

    sections.forEach((section) => {
        const input = section.querySelector('[data-search-input]');
        const button = section.querySelector('[data-search-button]');
        const statusEl = section.querySelector('[data-search-status]');

        if (!input) {
            return;
        }

        input.value = '';
        setSearchStatus(statusEl, '');

        let resetting = false;

        const performReset = async () => {
            if (!onIndexPage || resetting) {
                setSearchStatus(statusEl, '');
                updateCategoryQueryParam('');
                updateSearchQueryParam('');
                updateTagQueryParam('');
                return;
            }

            resetting = true;
            try {
                setSearchStatus(statusEl, '正在恢复列表…');
                updateCategoryQueryParam('');
                updateSearchQueryParam('');
                updateTagQueryParam('');
                await initIndexPage(rootPrefix);
                await typesetMath(listContainer);
                setSearchStatus(statusEl, '');
            } catch (error) {
                console.error('重置索引列表失败', error);
                setSearchStatus(statusEl, '未能恢复列表，请刷新页面。');
            } finally {
                resetting = false;
            }
        };

        const executeSearch = async () => {
            const query = input.value.trim();
            input.value = '';
            const activeQuery = getSearchQuery();
            if (!query) {
                if (activeQuery) {
                    await performReset();
                    setSearchStatus(statusEl, '');
                } else {
                    setSearchStatus(statusEl, '请输入关键词。');
                }
                return;
            }

            setSearchStatus(statusEl, '正在搜索…');

            let posts;
            try {
                posts = await loadAndCachePosts(rootPrefix);
            } catch (error) {
                console.error('Sidebar search manifest load failed', error);
                setSearchStatus(statusEl, '搜索失败，请稍后重试。');
                return;
            }

            if (onIndexPage) {
                updateCategoryQueryParam('');
                updateTagQueryParam('');
                updateSearchQueryParam(query);
                if (listContainer) {
                    listContainer.innerHTML = '<p class="post-list-placeholder">正在搜索相关文章…</p>';
                }
                const matches = await renderSearchResults(listContainer, paginationContainer, posts, query, rootPrefix);
                if (!matches.length) {
                    setSearchStatus(statusEl, `未找到与“${query}”相关的文章。`);
                } else {
                    setSearchStatus(statusEl, '');
                    try {
                        await typesetMath(listContainer);
                    } catch (error) {
                        console.error('搜索结果公式排版失败', error);
                    }
                }
                return;
            }

            const targetUrl = buildIndexSearchUrl(rootPrefix, query);
            setSearchStatus(statusEl, '正在跳转到首页显示搜索结果…');
            window.location.href = targetUrl;
        };

        button?.addEventListener('click', (event) => {
            event.preventDefault();
            executeSearch();
        });

        input.addEventListener('keydown', async (event) => {
            if (event.key === 'Enter') {
                event.preventDefault();
                await executeSearch();
            }
            if (event.key === 'Escape') {
                input.value = '';
                await performReset();
                setSearchStatus(statusEl, '');
            }
        });
    });
}

function renderPagination(container, currentPage, totalPages) {
    if (!container) {
        return;
    }

    if (totalPages <= 1) {
        container.innerHTML = '';
        container.setAttribute('hidden', '');
        return;
    }

    container.innerHTML = '';
    container.removeAttribute('hidden');

    if (currentPage > 1) {
        container.appendChild(createPaginationLink('‹', currentPage - 1, '上一页', 'page-num prev'));
    }

    const pages = buildPaginationSequence(currentPage, totalPages);
    let previousPage = 0;
    pages.forEach((page) => {
        if (previousPage && page - previousPage > 1) {
            container.appendChild(createEllipsis());
        }
        const isCurrent = page === currentPage;
        container.appendChild(createPaginationLink(String(page), page, `第 ${page} 页`, 'page-num', isCurrent));
        previousPage = page;
    });

    if (currentPage < totalPages) {
        container.appendChild(createPaginationLink('›', currentPage + 1, '下一页', 'page-num next'));
    }
}

function buildPaginationSequence(currentPage, totalPages) {
    const pages = new Set([1, totalPages]);
    for (let offset = -2; offset <= 2; offset += 1) {
        const candidate = currentPage + offset;
        if (candidate >= 1 && candidate <= totalPages) {
            pages.add(candidate);
        }
    }
    return Array.from(pages).sort((a, b) => a - b);
}

function createPaginationLink(label, page, ariaLabel, className, isCurrent = false) {
    const link = document.createElement('a');
    link.className = className;
    link.textContent = label;
    link.href = buildPageHref(page);
    link.setAttribute('aria-label', ariaLabel);
    if (isCurrent) {
        link.classList.add('active');
        link.setAttribute('aria-current', 'page');
    }
    return link;
}

async function ensureMathJax() {
    if (typeof window === 'undefined') {
        return null;
    }

    if (!mathJaxReadyPromise) {
        mathJaxReadyPromise = new Promise((resolve) => {
            const maxAttempts = 200; // ~10 seconds at 50ms intervals
            let attempts = 0;

            const poll = () => {
                const mathJax = window.MathJax;
                if (!mathJax) {
                    attempts += 1;
                    if (attempts >= maxAttempts) {
                        console.warn('MathJax script not detected within timeout.');
                        resolve(null);
                        return;
                    }
                    window.setTimeout(poll, 50);
                    return;
                }

                if (mathJax.startup && mathJax.startup.promise) {
                    mathJax.startup.promise
                        .then(() => resolve(window.MathJax))
                        .catch((error) => {
                            console.error('MathJax startup failed', error);
                            resolve(null);
                        });
                    return;
                }

                resolve(mathJax);
            };

            poll();
        });
    }

    try {
        return await mathJaxReadyPromise;
    } catch (error) {
        console.error('MathJax setup failed', error);
        return null;
    }
}

async function typesetMath(container) {
    const mathJax = await ensureMathJax();
    if (!mathJax) {
        return;
    }

    const targets = container ? (Array.isArray(container) ? container : [container]) : [document.body];

    if (typeof mathJax.typesetPromise === 'function') {
        try {
            await mathJax.typesetPromise(targets);
        } catch (error) {
            console.error('MathJax typeset failed', error);
        }
        return;
    }

    if (mathJax.Hub && typeof mathJax.Hub.Queue === 'function') {
        targets.forEach((target) => {
            mathJax.Hub.Queue(['Typeset', mathJax.Hub, target]);
        });
    }
}

function createEllipsis() {
    const span = document.createElement('span');
    span.className = 'page-ellipsis';
    span.textContent = '…';
    return span;
}

function buildPageHref(page) {
    const url = new URL(window.location.href);
    if (page <= 1) {
        url.searchParams.delete('page');
    } else {
        url.searchParams.set('page', String(page));
    }
    return `${url.pathname}${url.search}`;
}

function initContentOverview() {
    const tocSection = document.querySelector('.sidebar-section.toc');
    if (!tocSection) {
        return;
    }

    const tocList = tocSection.querySelector('.toc-list');
    const contentRoot = document.querySelector('.single-post-content');

    if (!tocList || !contentRoot) {
        tocSection.setAttribute('hidden', '');
        return;
    }

    const sections = Array.from(contentRoot.querySelectorAll('section[id]'));
    const entries = sections
        .map((section) => {
            const heading = section.querySelector('h1, h2, h3, h4, h5, h6');
            if (!heading) {
                return null;
            }

            const title = extractHeadingText(heading);
            if (!title) {
                return null;
            }

            return {
                id: section.id,
                title,
            };
        })
        .filter(Boolean);

    if (!entries.length) {
        tocSection.setAttribute('hidden', '');
        return;
    }

    tocSection.removeAttribute('hidden');
    tocList.innerHTML = '';

    entries.forEach(({ id, title }) => {
        const li = document.createElement('li');
        const link = document.createElement('a');
        link.href = `#${id}`;
        link.textContent = title;
        li.appendChild(link);
        tocList.appendChild(li);
    });
}

function extractHeadingText(heading) {
    const clone = heading.cloneNode(true);
    const anchors = clone.querySelectorAll('.section-anchor');
    anchors.forEach((anchor) => anchor.remove());
    const text = clone.textContent.trim();
    return text;
}

function computeRootPrefix() {
    if (typeof window === 'undefined') {
        return '';
    }

    const path = window.location.pathname.replace(/\\/g, '/');
    if (path.includes('/posts/')) {
        return '../';
    }
    return '';
}

function initSmoothScroll() {
    const links = document.querySelectorAll('a[href^="#"]');
    links.forEach((link) => {
        link.addEventListener('click', function (e) {
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                e.preventDefault();
                target.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });
}

function initBackToTop() {
    const backToTop = document.createElement('button');
    backToTop.className = 'back-to-top';
    backToTop.innerHTML = '↑';
    backToTop.style.cssText = `
        position: fixed;
        bottom: 30px;
        right: 30px;
        width: 40px;
        height: 40px;
        background-color: #4183c4;
        color: white;
        border: 1px solid #ddd;
        border-radius: 2px;
    font-size: 16px;
        cursor: pointer;
        opacity: 0;
        transition: opacity 0.3s ease;
        z-index: 999;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    `;

    document.body.appendChild(backToTop);

    window.addEventListener('scroll', function () {
        backToTop.style.opacity = window.scrollY > 300 ? '0.7' : '0';
    });

    backToTop.addEventListener('click', function () {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });

    backToTop.addEventListener('mouseenter', function () {
        this.style.opacity = '1';
    });

    backToTop.addEventListener('mouseleave', function () {
        if (window.scrollY > 300) {
            this.style.opacity = '0.7';
        }
    });
}

document.addEventListener('DOMContentLoaded', async () => {
    const rootPrefix = computeRootPrefix();
    const initialCategory = getCategoryFilter();
    const initialTag = getTagFilter();
    await injectPartials(rootPrefix);
    initFooterYear();
    initMobileDrawer(rootPrefix);

    // For non-index pages, prepare post navigation data before rendering components
    const listContainer = document.querySelector('[data-post-list]');
    if (!listContainer) {
        // Not on index page, initialize post navigation data first
        await initPostNavigation(rootPrefix);
    }

    await renderComponents(rootPrefix);
    fixPostMetaLinks(rootPrefix);

    // Initialize category nav and tag cloud (will be called again in initIndexPage if on index page)
    await initCategoryNav(rootPrefix, undefined, initialCategory);
    await initTagCloud(rootPrefix, undefined, initialTag);

    initSidebarSearch(rootPrefix);
    await initIndexPage(rootPrefix); // This handles random posts for index page

    // For non-index pages (like single post pages), initialize sidebar posts here
    if (!listContainer) {
        // Not on index page, so show recent posts in sidebar
        await initRecentPosts(rootPrefix);
    }

    const postList = document.querySelector('[data-post-list]');
    await typesetMath(postList || document.body);
    enhanceCodeBlocks(document.body);
    await highlightCodeBlocks(document.body);
    initSmoothScroll();
    initBackToTop();
    console.log('慢变量博客已加载完成！');
});

document.addEventListener('click', (event) => {
    handleCodeCopyClick(event);
});

async function initCategoryList(rootPrefix, postsArg) {
    const list = document.querySelector('[data-category-list]');
    if (!list) {
        return;
    }

    let posts = postsArg;
    if (!Array.isArray(posts)) {
        try {
            posts = await loadAndCachePosts(rootPrefix);
        } catch (error) {
            list.innerHTML = '';
            return;
        }
    }

    const counts = new Map();
    (posts || []).forEach((post) => {
        const categories = Array.isArray(post?.categories) ? post.categories : [];
        categories.forEach((category) => {
            const displayName = String(category || '').trim();
            if (!displayName) {
                return;
            }
            const key = normalizeCategoryValue(displayName);
            if (!counts.has(key)) {
                counts.set(key, { name: displayName, count: 0 });
            }
            counts.get(key).count += 1;
        });
    });

    const entries = Array.from(counts.values()).sort((a, b) => {
        if (b.count !== a.count) {
            return b.count - a.count;
        }
        return a.name.localeCompare(b.name, 'zh-Hans');
    });

    list.innerHTML = '';

    const appendItem = (label, value) => {
        const li = document.createElement('li');
        const a = document.createElement('a');
        a.href = buildCategoryHref(rootPrefix, value);
        a.textContent = label;
        a.addEventListener('click', async (event) => {
            event.preventDefault();
            const targetUrl = buildCategoryHref(rootPrefix, value);
            const listContainer = document.querySelector('[data-post-list]');
            if (listContainer) {
                updateCategoryQueryParam(value);
                updateSearchQueryParam('');
                await initIndexPage(rootPrefix);
                closeMobileDrawer();
                window.scrollTo({ top: 0, behavior: 'smooth' });
            } else {
                window.location.href = targetUrl;
            }
        });
        li.appendChild(a);
        list.appendChild(li);
    };

    appendItem('全部文章', '');
    entries.forEach(({ name }) => appendItem(name, name));
}

function initMobileDrawer(rootPrefix) {
    const drawer = document.getElementById('mobile-drawer');
    const backdrop = document.getElementById('mobile-drawer-backdrop');
    const btn = document.querySelector('[data-mobile-menu]');
    const closeBtn = document.querySelector('[data-drawer-close]');
    if (!drawer || !backdrop || !btn || !closeBtn) {
        return;
    }

    const open = () => {
        drawer.classList.add('open');
        drawer.removeAttribute('hidden');
        backdrop.removeAttribute('hidden');
        document.body.style.overflow = 'hidden';
    };

    const close = () => {
        drawer.classList.remove('open');
        drawer.setAttribute('hidden', '');
        backdrop.setAttribute('hidden', '');
        document.body.style.overflow = '';
    };

    btn.addEventListener('click', () => {
        open();
    });
    closeBtn.addEventListener('click', () => {
        close();
    });
    backdrop.addEventListener('click', () => {
        close();
    });

    initCategoryList(rootPrefix);
    initRecentPosts(rootPrefix);
}

function closeMobileDrawer() {
    const drawer = document.getElementById('mobile-drawer');
    const backdrop = document.getElementById('mobile-drawer-backdrop');
    if (!drawer || !backdrop) {
        return;
    }
    drawer.classList.remove('open');
    drawer.setAttribute('hidden', '');
    backdrop.setAttribute('hidden', '');
    document.body.style.overflow = '';
}
