#!/usr/bin/env node

const fs = require('fs/promises');
const path = require('path');

const ROOT_DIR = path.resolve(__dirname, '..');
const POSTS_DIR = path.join(ROOT_DIR, 'posts');
const MANIFEST_PATH = path.join(POSTS_DIR, 'manifest.json');

/**
 * Load the metadata JSON block embedded in a post HTML file.
 * @param {string} filePath
 * @param {string} fileName
 */
async function loadPostMetadata(filePath, fileName) {
    const rawHtml = await fs.readFile(filePath, 'utf-8');
    const metadataMatch = rawHtml.match(/<script\s+type="application\/json"\s+id="post-metadata">([\s\S]*?)<\/script>/i);

    if (!metadataMatch) {
        console.warn(`⚠️  Skipping ${fileName}: metadata block not found.`);
        return null;
    }

    let metadata;
    try {
        metadata = JSON.parse(metadataMatch[1].trim());
    } catch (error) {
        console.warn(`⚠️  Skipping ${fileName}: metadata JSON is invalid.`, error.message);
        return null;
    }

    const slug = metadata.slug || fileName.replace(/\.html$/, '');
    const createdDate = metadata.createdAt ? new Date(metadata.createdAt) : null;

    const derivedDay = createdDate ? String(createdDate.getDate()) : '';
    const derivedMonth = createdDate
        ? createdDate.toLocaleString('en-US', { month: 'short' })
        : '';

    const summary = extractSummaryBlocks(rawHtml, metadata.summary);

    return {
        id: metadata.id || slug,
        slug,
        title: metadata.title || slug,
        createdAt: metadata.createdAt || null,
        day: metadata.day || derivedDay,
        month: metadata.month || derivedMonth,
        categories: Array.isArray(metadata.categories) ? metadata.categories : [],
        tags: Array.isArray(metadata.tags) ? metadata.tags : [],
        summary,
        link: metadata.link || `posts/${slug}.html`,
        metaText: metadata.metaText || (metadata.createdAt ? `发布于 ${metadata.createdAt}` : ''),
    };
}

function extractSummaryBlocks(rawHtml, metadataSummary) {
    const markerBlocks = collectBlocksBeforeMarker(rawHtml);
    if (markerBlocks.length) {
        return markerBlocks;
    }

    if (Array.isArray(metadataSummary) && metadataSummary.length) {
        return metadataSummary;
    }

    return collectInitialBlocks(rawHtml, 2);
}

function collectBlocksBeforeMarker(rawHtml) {
    const markerMatch = rawHtml.match(/<!--\s*more\s*-->/i);
    if (!markerMatch) {
        return [];
    }

    const contentStart = rawHtml.indexOf('<div class="single-post-content"');
    if (contentStart === -1) {
        return [];
    }

    const openTagEnd = rawHtml.indexOf('>', contentStart);
    const markerIndex = markerMatch.index;
    if (openTagEnd === -1 || markerIndex <= openTagEnd) {
        return [];
    }

    const fragment = rawHtml.slice(openTagEnd + 1, markerIndex);
    return collectBlocks(fragment);
}

function collectInitialBlocks(rawHtml, limit) {
    const contentStart = rawHtml.indexOf('<div class="single-post-content"');
    if (contentStart === -1) {
        return [];
    }

    const openTagEnd = rawHtml.indexOf('>', contentStart);
    if (openTagEnd === -1) {
        return [];
    }

    const fragment = rawHtml.slice(openTagEnd + 1);
    return collectBlocks(fragment, limit);
}

function collectBlocks(fragment, limit = Infinity) {
    if (typeof fragment !== 'string' || !fragment.trim()) {
        return [];
    }

    const results = [];
    const blockRegex = /<(p|h2|h3|h4|h5|ul|ol|blockquote)([^>]*)>([\s\S]*?)<\/\1>/gi;
    let match;

    while ((match = blockRegex.exec(fragment)) && results.length < limit) {
        const type = match[1].toLowerCase();
        const innerHtml = (match[3] || '').trim();
        if (!innerHtml) {
            continue;
        }
        results.push({ type, html: innerHtml });
    }

    return results;
}

async function main() {
    const files = await fs.readdir(POSTS_DIR);
    const postFiles = files.filter((file) => file.endsWith('.html'));

    const posts = [];
    for (const file of postFiles) {
        const filePath = path.join(POSTS_DIR, file);
        const metadata = await loadPostMetadata(filePath, file);
        if (metadata) {
            posts.push(metadata);
        }
    }

    posts.sort((a, b) => {
        const aTime = a.createdAt ? new Date(a.createdAt).getTime() : 0;
        const bTime = b.createdAt ? new Date(b.createdAt).getTime() : 0;
        return bTime - aTime;
    });

    const manifest = {
        generatedAt: new Date().toISOString(),
        postCount: posts.length,
        posts,
    };

    await fs.writeFile(MANIFEST_PATH, `${JSON.stringify(manifest, null, 2)}\n`, 'utf-8');
    console.log(`✅ Generated manifest for ${posts.length} posts.`);
}

main().catch((error) => {
    console.error('❌ Failed to generate post manifest:', error);
    process.exit(1);
});
