#!/usr/bin/env node

const fs = require('fs/promises');
const path = require('path');
const { spawn } = require('child_process');
const matter = require('gray-matter');
const { marked } = require('marked');
const cheerio = require('cheerio');

const ROOT_DIR = path.resolve(__dirname, '..');
const POSTS_DIR = path.join(ROOT_DIR, 'posts');
const ASSETS_ROOT_DIR = path.join(ROOT_DIR, 'assets');
const MANIFEST_PATH = path.join(POSTS_DIR, 'manifest.json');

async function main() {
    try {
        const { sourcePath, options } = parseArgs(process.argv.slice(2));
        if (options.mode === 'delete') {
            await handleDelete(options);
            if (options.manifest !== false) {
                await runManifestGenerator();
                console.log('✅ Updated posts/manifest.json');
            }
            return;
        }

        if (!sourcePath) {
            printUsage();
            process.exit(1);
        }

        const absoluteMarkdownPath = path.resolve(process.cwd(), sourcePath);
        await assertFileExists(absoluteMarkdownPath, `Markdown source not found: ${absoluteMarkdownPath}`);

        const rawMarkdown = await fs.readFile(absoluteMarkdownPath, 'utf-8');
        const { content: markdownBody, data: frontmatter } = matter(rawMarkdown);

        if (frontmatter.draft === true) {
            throw new Error('Front matter sets "draft: true". Refusing to publish.');
        }

        const title = normalizeTitle(options.title ?? frontmatter.title);
        const createdAt = normalizeDate(frontmatter.createdAt ?? frontmatter.date);
        const createdDate = new Date(createdAt);

        const categories = normalizeStringArray(frontmatter.categories);
        const tags = normalizeStringArray(frontmatter.tags);
        const summary = normalizeSummary(frontmatter.summary);

        const manifestEntries = await loadManifestEntries();
        const slug = await resolveSlug({
            slug: options.slug ?? frontmatter.slug,
            title,
            sourcePath: absoluteMarkdownPath,
            fallbackId: options.id ?? frontmatter.id,
        });

        const slugCandidates = Array.from(
            new Set([
                options.slug,
                frontmatter.slug,
                slug,
                path.basename(sourcePath, path.extname(sourcePath)),
            ].filter(Boolean)),
        );

        let existingEntry = null;
        if (options.mode === 'update') {
            existingEntry = findExistingPost(manifestEntries, {
                id: options.id ?? frontmatter.id,
                slugCandidates,
            });
            if (!existingEntry) {
                throw new Error('No existing post found to update. Provide the correct "id" or "slug".');
            }
        } else {
            ensureCreateTargetAvailable(manifestEntries, {
                idCandidate: options.id ?? frontmatter.id,
                slugCandidate: slug,
            });
        }

        let id;
        if (options.mode === 'create') {
            id = await resolveId(options.id ?? frontmatter.id);
        } else {
            id = String(options.id ?? frontmatter.id ?? existingEntry.id ?? '').trim();
            if (!id) {
                throw new Error('Unable to determine post "id" for update. Add "id" to the front matter or pass --id.');
            }
            const idConflict = manifestEntries.find((entry) => String(entry.id) === id && entry.slug !== existingEntry.slug);
            if (idConflict) {
                throw new Error(`Cannot update: desired id "${id}" is already used by post "${idConflict.slug}".`);
            }
        }

        const outputFileName = `${slug}.html`;
        const outputPath = path.join(POSTS_DIR, outputFileName);

        if (options.mode === 'update' && existingEntry) {
            const slugConflict = manifestEntries.find((entry) => entry.slug === slug && entry.slug !== existingEntry.slug);
            if (slugConflict) {
                throw new Error(`Cannot update: desired slug "${slug}" is already used by another post.`);
            }
        }

        if (options.mode === 'create') {
            await assertFileNotExists(outputPath, `Post output already exists at posts/${outputFileName}`);
        }

        let previousOutputPath = '';
        let slugChanged = false;
        if (options.mode === 'update' && existingEntry) {
            previousOutputPath = path.join(POSTS_DIR, `${existingEntry.slug}.html`);
            slugChanged = existingEntry.slug !== slug;
            if (!slugChanged) {
                const exists = await fileExists(previousOutputPath);
                if (!exists) {
                    throw new Error(`Cannot update: expected post file missing at posts/${existingEntry.slug}.html`);
                }
            }
        }

        if (options.mode === 'update' && slugChanged) {
            const previousExists = await fileExists(previousOutputPath);
            if (!previousExists) {
                throw new Error(`Cannot update: original post file not found at posts/${existingEntry.slug}.html`);
            }
        }

        const targetAssetsDir = path.join(ASSETS_ROOT_DIR, id);
        if (options.mode === 'update') {
            if (existingEntry && existingEntry.id && existingEntry.id !== id) {
                await removeDirIfExists(path.join(ASSETS_ROOT_DIR, existingEntry.id));
            }
            await removeDirIfExists(targetAssetsDir);
        }

        const assetsMap = await processMarkdownAssets({
            markdown: markdownBody,
            markdownPath: absoluteMarkdownPath,
            postId: id,
        });

        const metadata = {
            id,
            slug,
            title,
            createdAt,
            day: String(createdDate.getDate()),
            month: createdDate.toLocaleString('en-US', { month: 'short' }),
            categories,
            tags,
            summary,
            link: `posts/${slug}.html`,
        };

        const metadataJson = JSON.stringify(metadata, null, 4);
    let bodyHtml = renderMarkdown(markdownBody, assetsMap);
    bodyHtml = rewriteInlineHtmlImages(bodyHtml, assetsMap);
    bodyHtml = wrapPostContentSections(bodyHtml);
        const metaHtml = buildMetaHtml(createdAt, categories, tags);

        const pageHtml = buildPageHtml({
            title,
            metadataJson,
            bodyHtml,
            day: metadata.day,
            month: metadata.month,
            metaHtml,
        });

        await fs.writeFile(outputPath, `${pageHtml}\n`, 'utf-8');

        if (options.mode === 'create') {
            console.log(`✅ Created post: posts/${outputFileName}`);
        } else {
            console.log(`✅ Updated post: posts/${outputFileName}`);
            if (slugChanged) {
                await removeFileIfExists(previousOutputPath);
                console.log(`ℹ️ Removed previous file: posts/${existingEntry.slug}.html`);
            }
        }

        if (options.manifest !== false) {
            await runManifestGenerator();
            console.log('✅ Updated posts/manifest.json');
        }
    } catch (error) {
        console.error(`❌ ${error.message}`);
        process.exit(1);
    }
}

function parseArgs(args) {
    let sourcePath = '';
    const options = { manifest: true, mode: 'create' };

    for (let index = 0; index < args.length; index += 1) {
        const arg = args[index];
        if (arg.startsWith('--')) {
            const { key, value } = parseFlag(arg, args[index + 1]);
            if (value === undefined && key !== 'manifest' && key !== 'no-manifest') {
                throw new Error(`Flag "--${key}" requires a value.`);
            }

            switch (key) {
                case 'slug':
                case 'id':
                case 'title':
                    options[key] = value;
                    break;
                case 'mode':
                    options.mode = normalizeMode(value);
                    break;
                case 'no-manifest':
                    options.manifest = false;
                    break;
                case 'manifest':
                    options.manifest = value !== 'false';
                    if (value === undefined) {
                        options.manifest = true;
                    }
                    break;
                default:
                    throw new Error(`Unknown flag: --${key}`);
            }

            if (!arg.includes('=') && (key === 'slug' || key === 'id' || key === 'title' || key === 'manifest' || key === 'mode')) {
                index += 1;
            }
        } else if (!sourcePath) {
            sourcePath = arg;
        } else {
            throw new Error(`Unexpected argument: ${arg}`);
        }
    }

    return { sourcePath, options };
}

function parseFlag(flag, nextArg) {
    const trimmed = flag.replace(/^--/, '');
    if (trimmed === 'no-manifest') {
        return { key: 'no-manifest', value: false };
    }

    const [keyPart, valuePart] = trimmed.split('=');
    if (valuePart !== undefined) {
        return { key: keyPart, value: valuePart };
    }

    if (nextArg && !nextArg.startsWith('--')) {
        return { key: keyPart, value: nextArg };
    }

    return { key: keyPart, value: undefined };
}

function normalizeMode(value) {
    const normalized = String(value ?? '').trim().toLowerCase();
    if (normalized === 'create' || normalized === '') {
        return 'create';
    }
    if (normalized === 'update') {
        return 'update';
    }
    if (normalized === 'delete') {
        return 'delete';
    }
    throw new Error(`Unsupported mode: ${value}. Expected "create", "update", or "delete".`);
}

async function assertFileExists(filePath, notFoundMessage) {
    try {
        const stat = await fs.stat(filePath);
        if (!stat.isFile()) {
            throw new Error();
        }
    } catch {
        throw new Error(notFoundMessage || `Markdown source not found: ${filePath}`);
    }
}

async function assertFileNotExists(filePath, message) {
    try {
        await fs.access(filePath);
        throw new Error(message);
    } catch (error) {
        if (error && error.code === 'ENOENT') {
            return;
        }
        throw error;
    }
}

async function fileExists(filePath) {
    try {
        await fs.access(filePath);
        return true;
    } catch (error) {
        if (error && error.code === 'ENOENT') {
            return false;
        }
        throw error;
    }
}

async function removeFileIfExists(filePath) {
    try {
        await fs.rm(filePath, { force: true });
    } catch (error) {
        if (error && error.code === 'ENOENT') {
            return;
        }
        throw error;
    }
}

async function removeDirIfExists(dirPath, { silent = false, label } = {}) {
    try {
        const stat = await fs.stat(dirPath);
        if (!stat.isDirectory()) {
            return;
        }
    } catch (error) {
        if (error && error.code === 'ENOENT') {
            return;
        }
        throw error;
    }

    await fs.rm(dirPath, { recursive: true, force: true });
    if (!silent) {
        const display = label || toPosix(path.relative(ROOT_DIR, dirPath));
        console.log(`ℹ️ Removed ${display}`);
    }
}

async function loadManifestEntries() {
    try {
        const raw = await fs.readFile(MANIFEST_PATH, 'utf-8');
        const parsed = JSON.parse(raw);
        if (parsed && Array.isArray(parsed.posts)) {
            return parsed.posts;
        }
    } catch (error) {
        if (error && error.code !== 'ENOENT') {
            throw error;
        }
    }
    return [];
}

function findExistingPost(entries, { id, slugCandidates = [] }) {
    if (!Array.isArray(entries) || !entries.length) {
        return null;
    }

    const normalizedId = typeof id === 'undefined' || id === null ? '' : String(id).trim();
    if (normalizedId) {
        const matchById = entries.find((entry) => String(entry.id) === normalizedId);
        if (matchById) {
            return matchById;
        }
    }

    for (const candidate of slugCandidates) {
        const normalized = typeof candidate === 'string' ? candidate.trim() : '';
        if (!normalized) {
            continue;
        }
        const matchBySlug = entries.find((entry) => entry.slug === normalized);
        if (matchBySlug) {
            return matchBySlug;
        }
    }

    return null;
}

function ensureCreateTargetAvailable(entries, { idCandidate, slugCandidate }) {
    if (!Array.isArray(entries) || !entries.length) {
        return;
    }

    const normalizedId = typeof idCandidate === 'undefined' || idCandidate === null ? '' : String(idCandidate).trim();
    if (normalizedId) {
        const conflictById = entries.find((entry) => String(entry.id) === normalizedId);
        if (conflictById) {
            throw new Error(`Post id "${normalizedId}" already exists (slug: ${conflictById.slug}). Use --mode update to modify it.`);
        }
    }

    const normalizedSlug = typeof slugCandidate === 'string' ? slugCandidate.trim() : '';
    if (normalizedSlug) {
        const conflictBySlug = entries.find((entry) => entry.slug === normalizedSlug);
        if (conflictBySlug) {
            throw new Error(`Post slug "${normalizedSlug}" already exists. Use --mode update to modify it.`);
        }
    }
}

async function handleDelete(options) {
    const entries = await loadManifestEntries();
    const slugCandidates = [options.slug].filter(Boolean);
    const entry = findExistingPost(entries, {
        id: options.id,
        slugCandidates,
    });

    const resolvedSlug = entry ? entry.slug : (typeof options.slug === 'string' ? options.slug.trim() : '');
    if (!resolvedSlug) {
        throw new Error('Delete mode requires a known post. Provide --slug or --id referring to an existing post.');
    }

    const postPath = path.join(POSTS_DIR, `${resolvedSlug}.html`);
    const exists = await fileExists(postPath);
    if (!exists) {
        throw new Error(`Post file not found at posts/${resolvedSlug}.html`);
    }

    await fs.rm(postPath, { force: true });
    console.log(`✅ Deleted post: posts/${resolvedSlug}.html`);

    const assetId = entry ? entry.id : options.id;
    if (assetId) {
        await removeDirIfExists(path.join(ASSETS_ROOT_DIR, String(assetId)), { label: `assets/${assetId}` });
    }
}

function normalizeTitle(value) {
    const title = typeof value === 'string' ? value.trim() : '';
    if (!title) {
        throw new Error('Front matter must include a non-empty "title".');
    }
    return title;
}

function normalizeDate(value) {
    if (!value) {
        throw new Error('Front matter must include "createdAt" (ISO date).');
    }
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
        throw new Error(`Invalid "createdAt" value: ${value}`);
    }
    return date.toISOString().split('T')[0];
}

async function resolveId(value) {
    if (value) {
        return String(value).trim();
    }

    const candidates = [];

    try {
        const manifestRaw = await fs.readFile(MANIFEST_PATH, 'utf-8');
        const manifest = JSON.parse(manifestRaw);
        const fromManifest = (manifest.posts || [])
            .map((post) => Number.parseInt(post.id, 10))
            .filter((num) => Number.isFinite(num));
        candidates.push(...fromManifest);
    } catch (_) {
        // Manifest missing, fall back to scanning post filenames.
    }

    try {
        const files = await fs.readdir(POSTS_DIR);
        files.forEach((file) => {
            if (!file.endsWith('.html')) {
                return;
            }
            const numeric = Number.parseInt(file.replace(/\.html$/i, ''), 10);
            if (Number.isFinite(numeric)) {
                candidates.push(numeric);
            }
        });
    } catch (_) {
        // Ignore, directory should exist.
    }

    const currentMax = candidates.length ? Math.max(...candidates) : 11384;
    return String(currentMax + 1);
}

async function resolveSlug({ slug, title, sourcePath, fallbackId }) {
    if (slug) {
        return String(slug).trim();
    }

    const candidates = [title, path.basename(sourcePath, path.extname(sourcePath)), fallbackId];
    for (const candidate of candidates) {
        const normalized = slugify(candidate);
        if (normalized) {
            return normalized;
        }
    }

    throw new Error('Unable to derive a slug. Please provide "slug" in front matter or via --slug.');
}

function slugify(value) {
    if (typeof value !== 'string') {
        return '';
    }
    return value
        .trim()
        .toLowerCase()
        .replace(/[^a-z0-9\u4e00-\u9fa5\-\s_]/g, '')
        .replace(/\s+/g, '-')
        .replace(/-+/g, '-')
        .replace(/^[-_]+|[-_]+$/g, '')
        .replace(/[\u4e00-\u9fa5]/g, '') // drop Han characters to avoid filesystem surprises
        .trim();
}

function normalizeStringArray(value) {
    if (Array.isArray(value)) {
        return value
            .map((item) => (typeof item === 'string' ? item.trim() : ''))
            .filter(Boolean);
    }
    if (typeof value === 'string') {
        return value
            .split(/[，,]/)
            .map((item) => item.trim())
            .filter(Boolean);
    }
    return [];
}

function normalizeSummary(value) {
    if (!value) {
        return [];
    }
    if (Array.isArray(value)) {
        return value
            .map((block) => {
                if (!block || typeof block !== 'object') {
                    return null;
                }
                const type = typeof block.type === 'string' ? block.type.trim() : '';
                const html = typeof block.html === 'string' ? block.html.trim() : '';
                if (!type || !html) {
                    return null;
                }
                return { type, html };
            })
            .filter(Boolean);
    }
    if (typeof value === 'string') {
        return [{ type: 'p', html: escapeHtml(value) }];
    }
    return [];
}

function renderMarkdown(markdown, assetsMap = new Map()) {
    const renderer = new marked.Renderer();
    const slugger = createHeadingSlugger();

    renderer.heading = (text, level, raw) => {
        const slug = slugger.slug(raw);
        const anchor = level >= 2 && level <= 4 ? ' <span class="section-anchor">#</span>' : '';
        return `<h${level} id="${slug}">${text}${anchor}</h${level}>\n`;
    };

    renderer.image = (href, title, text) => {
        const resolvedHref = resolveAssetHref(href, assetsMap);
        const alt = escapeAttribute(text || '');
        const titleAttr = title ? ` title="${escapeAttribute(title)}"` : '';
        return `<img src="${escapeAttribute(resolvedHref)}" alt="${alt}"${titleAttr}>`;
    };

    return marked.parse(markdown, {
        gfm: true,
        breaks: false,
        smartLists: true,
        smartypants: false,
        renderer,
        headerIds: true,
        mangle: false,
    }).trim();
}

function buildMetaHtml(createdAt, categories, tags) {
    const segments = [];
    segments.push(`<span class="meta-item meta-date">${escapeHtml(createdAt)}</span>`);
    segments.push('<span class="meta-divider">|</span>');

    if (categories.length) {
        const categoryHtml = categories
            .map((category) => `<a href="#">${escapeHtml(category)}</a>`)
            .join(',');
        segments.push(`<span class="meta-item meta-categories">分类：${categoryHtml}</span>`);
    } else {
        segments.push('<span class="meta-item meta-categories">分类：<span class="post-taxonomy-empty">未分类</span></span>');
    }

    segments.push('<span class="meta-divider">|</span>');

    if (tags.length) {
        const tagHtml = tags
            .map((tag) => `<a href="#">${escapeHtml(tag)}</a>`)
            .join(',');
        segments.push(`<span class="meta-item meta-tags">标签：${tagHtml}</span>`);
    } else {
        segments.push('<span class="meta-item meta-tags">标签：<span class="post-taxonomy-empty">暂无标签</span></span>');
    }

    return segments.join('');
}

function buildPageHtml({ title, metadataJson, bodyHtml, day, month, metaHtml }) {
    const metadataBlock = indentBlock(metadataJson, 8);
    const contentBlock = indentBlock(bodyHtml, 16);

    return `<!DOCTYPE html>
<html lang="zh-CN">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${escapeHtml(title)} | 科学空间</title>
    <link rel="stylesheet" href="../styles.css">

    <script type="application/json" id="post-metadata">
${metadataBlock}
    </script>

    <script>
        MathJax = {
            tex: {
                inlineMath: [['$', '$'], ['\\(', '\\)']],
                displayMath: [['$$', '$$'], ['\\[', '\\]']],
                processEscapes: true,
                processEnvironments: true
            },
            options: {
                skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre']
            }
        };
    </script>
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" id="MathJax-script" async></script>
</head>

<body>
    <div class="page-container">
        <div data-include="../partials/navbar.html"></div>

    <main class="main-content">
        <div class="wrapper">
            <article class="post">
                <div data-component="post-header"
                     data-day="${escapeAttribute(day)}"
                     data-month="${escapeAttribute(month)}"
                     data-title="${escapeAttribute(title)}"
                     data-link=""
                     data-meta="${escapeAttribute(metaHtml)}"
                     data-heading="h1"
                     data-meta-class="post-meta"
                     data-title-class="post-title"></div>

                <div class="post-content single-post-content">
${contentBlock}
                </div>

                <div data-component="post-navigation"></div>
            </article>
        </div>

        <div data-include="../partials/sidebar-post.html"></div>
    </main>

    <div data-include="../partials/footer.html"></div>
    </div><!-- end page-container -->

    <script src="../script.js"></script>
</body>

</html>`;
}

function indentBlock(text, spaces) {
    const indent = ' '.repeat(spaces);
    const normalized = text.replace(/\r\n/g, '\n').trim();
    return normalized
        .split('\n')
        .map((line) => `${indent}${line}`)
        .join('\n');
}

function escapeHtml(value) {
    return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
}

function escapeAttribute(value) {
    return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function createHeadingSlugger() {
    const counts = new Map();
    return {
        slug(value) {
            const base = slugifyHeading(value);
            const seen = counts.get(base) || 0;
            counts.set(base, seen + 1);
            if (seen === 0) {
                return base;
            }
            return `${base}-${seen}`;
        },
    };
}

function slugifyHeading(value) {
    if (typeof value !== 'string') {
        return 'section';
    }

    const normalized = value
        .trim()
        .toLowerCase()
        .replace(/\s+/g, '-')
        .replace(/[!"#$%&'()*+,./:;<=>?@[\]^`{|}~]/g, '')
        .replace(/-+/g, '-')
        .replace(/^-+|-+$/g, '');

    return normalized || 'section';
}

function resolveAssetHref(href, assetsMap) {
    if (!href) {
        return href;
    }

    const lookupSequence = [];
    const trimmed = href.trim();
    if (trimmed) {
        lookupSequence.push(trimmed);
    }

    const canonical = canonicalizeAssetPath(trimmed || href, { stripLeadingDot: true });
    if (canonical && canonical !== trimmed) {
        lookupSequence.push(canonical);
    }

    for (const key of lookupSequence) {
        const candidate = getFromAssetMap(assetsMap, key);
        if (candidate) {
            return candidate;
        }
    }

    return href;
}

function rewriteInlineHtmlImages(html, assetsMap) {
    if (!html || !assetsMap || (assetsMap instanceof Map && assetsMap.size === 0)) {
        return html;
    }

    const mapSize = assetsMap instanceof Map ? assetsMap.size : Object.keys(assetsMap || {}).length;
    if (!mapSize) {
        return html;
    }

    const imgTagPattern = /<img\b[^>]*>/gi;
    return html.replace(imgTagPattern, (tag) => {
        const quotedSrcPattern = /\bsrc\s*=\s*("|')([^"']+)(\1)/i;
        const unquotedSrcPattern = /\bsrc\s*=\s*([^\s"'>]+)/i;

        if (quotedSrcPattern.test(tag)) {
            return tag.replace(quotedSrcPattern, (match, quote, value) => {
                const resolved = resolveAssetHref(value, assetsMap);
                if (resolved === value) {
                    return match;
                }
                return `src=${quote}${escapeAttribute(resolved)}${quote}`;
            });
        }

        if (unquotedSrcPattern.test(tag)) {
            return tag.replace(unquotedSrcPattern, (match, value) => {
                const resolved = resolveAssetHref(value, assetsMap);
                if (resolved === value) {
                    return match;
                }
                return `src="${escapeAttribute(resolved)}"`;
            });
        }

        return tag;
    });
}

function wrapPostContentSections(html) {
    if (typeof html !== 'string' || !html.trim()) {
        return html;
    }

    const $ = cheerio.load(`<div id="__post-root">${html}</div>`, {
        decodeEntities: false,
    });

    const root = $('#__post-root');
    const container = root.get(0);
    if (!container || !container.firstChild) {
        return html;
    }

    let node = container.firstChild;
    while (node) {
        const nextNode = node.nextSibling;
        if (node.type === 'tag' && node.name === 'h2') {
            const heading = $(node);

            if (heading.parents('section').length) {
                node = nextNode;
                continue;
            }

            const headingId = (heading.attr('id') || '').trim();
            const fallbackId = slugifyHeading(extractHeadingPlainText(heading));
            const sectionId = headingId || fallbackId;

            if (!sectionId) {
                node = nextNode;
                continue;
            }

            const section = $('<section></section>');
            section.addClass('post-section');
            section.attr('id', sectionId);

            heading.removeAttr('id');
            heading.attr('data-heading-id', sectionId);

            heading.before(section);
            section.append(heading);

            let siblingNode = section.get(0).nextSibling;
            while (siblingNode) {
                if (siblingNode.type === 'tag' && siblingNode.name === 'h2') {
                    break;
                }
                const nextSibling = siblingNode.nextSibling;
                section.append($(siblingNode));
                siblingNode = nextSibling;
            }
        }

        node = nextNode;
    }

    return root.html().trim();
}

function extractHeadingPlainText(heading) {
    const clone = heading.clone();
    clone.find('.section-anchor').remove();
    const text = clone.text().replace(/#/g, '').trim();
    return text;
}

async function processMarkdownAssets({ markdown, markdownPath, postId }) {
    const sources = collectMarkdownImageSources(markdown);
    const localSources = sources.filter((src) => isLocalAssetPath(src));

    if (!localSources.length) {
        return new Map();
    }

    await fs.mkdir(ASSETS_ROOT_DIR, { recursive: true });

    const postAssetsDir = path.join(ASSETS_ROOT_DIR, postId);
    await assertAssetsDirectoryAvailable(postAssetsDir);

    const assetMap = new Map();
    const canonicalCache = new Map();
    const usedNames = new Set();
    const markdownDir = path.dirname(markdownPath);

    for (const rawHref of localSources) {
        const canonical = canonicalizeAssetPath(rawHref);
        const absoluteSource = path.resolve(markdownDir, canonical);
        await assertFileExists(
            absoluteSource,
            `Image not found for markdown reference "${rawHref}" (resolved to ${absoluteSource})`,
        );

        let relativeFromPosts = canonicalCache.get(absoluteSource);
        if (!relativeFromPosts) {
            const desiredName = sanitizeAssetFileName(path.basename(canonical));
            const uniqueName = ensureUniqueFileName(desiredName, usedNames);
            const destinationPath = path.join(postAssetsDir, uniqueName);

            await fs.copyFile(absoluteSource, destinationPath);

            usedNames.add(uniqueName);
            relativeFromPosts = toPosix(path.relative(POSTS_DIR, destinationPath));
            canonicalCache.set(absoluteSource, relativeFromPosts);
        }

        assetMap.set(rawHref, relativeFromPosts);

        const normalizedKey = canonicalizeAssetPath(rawHref, { stripLeadingDot: true });
        if (normalizedKey && normalizedKey !== rawHref) {
            assetMap.set(normalizedKey, relativeFromPosts);
        }
    }

    console.log(`✅ Copied ${canonicalCache.size} image${canonicalCache.size === 1 ? '' : 's'} to assets/${postId}`);

    return assetMap;
}

function collectMarkdownImageSources(markdown) {
    const results = [];
    const seen = new Set();
    const tokens = marked.lexer(markdown, { gfm: true });

    marked.walkTokens(tokens, (token) => {
        if (!token) {
            return;
        }

        if (token.type === 'image' && typeof token.href === 'string') {
            const value = token.href.trim();
            if (value && !seen.has(value)) {
                results.push(value);
                seen.add(value);
            }
            return;
        }

        if (token.type === 'html') {
            const html = typeof token.text === 'string' ? token.text : typeof token.raw === 'string' ? token.raw : '';
            if (!html) {
                return;
            }

            const htmlSources = extractHtmlImageSources(html);
            htmlSources.forEach((value) => {
                if (value && !seen.has(value)) {
                    results.push(value);
                    seen.add(value);
                }
            });
        }
    });

    return results;
}

function extractHtmlImageSources(html) {
    const sources = [];
    const pattern = /<img\b[^>]*?\bsrc\s*=\s*("([^"]+)"|'([^']+)'|([^"'\s>]+))/gi;
    let match;
    while ((match = pattern.exec(html))) {
        const value = match[2] || match[3] || match[4];
        if (value) {
            sources.push(value.trim());
        }
    }
    return sources;
}

function isLocalAssetPath(href) {
    if (!href) {
        return false;
    }

    const trimmed = href.trim();
    if (!trimmed) {
        return false;
    }

    if (/^[a-z][a-z0-9+.-]*:/i.test(trimmed)) {
        return false;
    }

    if (/^\/\//.test(trimmed)) {
        return false;
    }

    if (trimmed.startsWith('#')) {
        return false;
    }

    if (path.isAbsolute(trimmed)) {
        return false;
    }

    return true;
}

function canonicalizeAssetPath(value, { stripLeadingDot = false } = {}) {
    if (typeof value !== 'string') {
        return '';
    }

    let normalized = value.trim();
    if (!normalized) {
        return '';
    }

    try {
        normalized = decodeURI(normalized);
    } catch (_) {
        // Ignore decode errors and fall back to original string.
    }

    normalized = normalized.replace(/\\/g, '/');
    if (stripLeadingDot) {
        normalized = normalized.replace(/^\.\//, '');
    }

    return normalized;
}

async function assertAssetsDirectoryAvailable(dirPath) {
    try {
        const stat = await fs.stat(dirPath);
        if (stat.isDirectory()) {
            throw new Error(`Assets directory already exists: ${dirPath}`);
        }
        throw new Error(`Path already exists: ${dirPath}`);
    } catch (error) {
        if (error && error.code === 'ENOENT') {
            await fs.mkdir(dirPath, { recursive: true });
            return;
        }
        throw error;
    }
}

function sanitizeAssetFileName(fileName) {
    const trimmed = (fileName || '').trim();
    if (!trimmed) {
        return 'asset';
    }

    const invalidCharsPattern = /[\s<>:"/\\|?*]+/g;
    const sanitized = trimmed.replace(invalidCharsPattern, '-').replace(/-+/g, '-').replace(/^-+|-+$/g, '');
    return sanitized || 'asset';
}

function ensureUniqueFileName(fileName, usedNames) {
    const ext = path.extname(fileName);
    const base = ext ? fileName.slice(0, -ext.length) : fileName;
    let candidate = fileName;
    let index = 1;

    while (usedNames.has(candidate)) {
        candidate = `${base}-${index}${ext}`;
        index += 1;
    }

    return candidate;
}

function getFromAssetMap(map, key) {
    if (!key) {
        return undefined;
    }

    if (map instanceof Map) {
        return map.get(key);
    }

    if (map && typeof map === 'object') {
        return map[key];
    }

    return undefined;
}

function toPosix(value) {
    return value.replace(/\\/g, '/');
}

async function runManifestGenerator() {
    await new Promise((resolve, reject) => {
        const child = spawn(process.execPath, ['scripts/generate-post-manifest.js'], {
            cwd: ROOT_DIR,
            stdio: 'inherit',
        });
        child.on('close', (code) => {
            if (code === 0) {
                resolve();
            } else {
                reject(new Error(`Manifest generator exited with code ${code}`));
            }
        });
        child.on('error', (error) => {
            reject(error);
        });
    });
}

function printUsage() {
    console.log('Usage:');
    console.log('  Create: node scripts/create-post-from-markdown.js <markdown-path> [--slug <slug>] [--id <id>] [--title <title>] [--mode create] [--no-manifest]');
    console.log('  Update: node scripts/create-post-from-markdown.js <markdown-path> --mode update [--slug <slug>] [--id <id>] [--title <title>]');
    console.log('  Delete: node scripts/create-post-from-markdown.js --mode delete --slug <slug> [--id <id>] [--no-manifest]');
}

main();
