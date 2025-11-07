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
        metaText: buildMetaText(metadata),
    };
}

function buildMetaText(metadata) {
    if (!metadata) {
        return '';
    }

    const segments = [];

    if (metadata.createdAt) {
        segments.push(String(metadata.createdAt));
    }

    const categories = Array.isArray(metadata.categories) ? metadata.categories.filter(Boolean) : [];
        const categoryText = categories.length ? categories.join(',') : '未分类';
    segments.push(`分类：${categoryText}`);

    const tags = Array.isArray(metadata.tags) ? metadata.tags.filter(Boolean) : [];
        const tagText = tags.length ? tags.join(',') : '暂无标签';
    segments.push(`标签：${tagText}`);

    return segments.join(' | ');
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

    // Look for either "post-content single-post-content" or just "post-content"
    const contentStart = rawHtml.search(/<div[^>]*class="[^"]*post-content[^"]*"/i);
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
    const contentStart = rawHtml.search(/<div[^>]*class="[^"]*post-content[^"]*"/i);
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
    
    // Remove section tags but keep their content
    // Use a simple approach: repeatedly remove section tags
    let workingFragment = fragment;
    let prevLength = 0;
    while (workingFragment.length !== prevLength) {
        prevLength = workingFragment.length;
        workingFragment = workingFragment.replace(/<section[^>]*>/gi, '').replace(/<\/section>/gi, '');
    }
    
    const blockRegex = /<(p|h2|h3|h4|h5|ul|ol|blockquote|figure|table|pre)([^>]*)>([\s\S]*?)<\/\1>/gi;
    let match;

    while ((match = blockRegex.exec(workingFragment)) && results.length < limit) {
        const type = match[1].toLowerCase();
        const attributes = (match[2] || '').trim();
        const innerHtml = (match[3] || '').trim();
        
        if (!innerHtml) {
            continue;
        }

        // For complex elements (table, ul, ol, blockquote, figure, pre), 
        // preserve the outer tag to maintain structure
        const needsOuterTag = ['table', 'ul', 'ol', 'blockquote', 'figure', 'pre'].includes(type);
        
        let html;
        if (needsOuterTag) {
            // Reconstruct the complete element with its tag and attributes
            const attrString = attributes ? ` ${attributes}` : '';
            html = `<${type}${attrString}>${innerHtml}</${type}>`;
        } else {
            // For simple elements (p, h2-h5), just use the inner content
            html = innerHtml;
        }
        
        results.push({ type, html });
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
