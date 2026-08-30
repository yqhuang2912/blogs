#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const POSTS = path.join(ROOT, 'posts');
const errors = [];

function listFiles(directory, extension) {
    return fs.readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
        const target = path.join(directory, entry.name);
        if (entry.isDirectory()) {
            return listFiles(target, extension);
        }
        return !extension || target.endsWith(extension) ? [target] : [];
    });
}

const htmlFiles = [
    path.join(ROOT, 'index.html'),
    ...listFiles(POSTS, '.html'),
    ...listFiles(path.join(ROOT, 'partials'), '.html'),
];

for (const file of htmlFiles) {
    const html = fs.readFileSync(file, 'utf8');
    const references = html.matchAll(/\b(?:href|src)\s*=\s*["']([^"']+)["']/gi);
    for (const [, reference] of references) {
        if (!reference
            || /^(?:https?:|mailto:|data:|#|\/\/)/i.test(reference)
            || reference.includes('{{ROOT}}')) {
            continue;
        }
        const cleanReference = reference.split(/[?#]/, 1)[0];
        const target = path.resolve(path.dirname(file), cleanReference);
        if (!fs.existsSync(target)) {
            errors.push(`${path.relative(ROOT, file)} references missing ${reference}`);
        }
    }
}

const manifest = JSON.parse(fs.readFileSync(path.join(POSTS, 'manifest.json'), 'utf8'));
const postSlugs = fs.readdirSync(POSTS)
    .filter((file) => file.endsWith('.html'))
    .map((file) => file.slice(0, -5));
const manifestSlugs = manifest.posts.map((post) => post.slug);

for (const slug of postSlugs.filter((value) => !manifestSlugs.includes(value))) {
    errors.push(`posts/${slug}.html is missing from posts/manifest.json`);
}
for (const slug of manifestSlugs.filter((value) => !postSlugs.includes(value))) {
    errors.push(`manifest entry ${slug} has no HTML file`);
}
for (const field of ['id', 'slug']) {
    const values = manifest.posts.map((post) => String(post[field]));
    for (const duplicate of values.filter((value, index) => values.indexOf(value) !== index)) {
        errors.push(`duplicate post ${field}: ${duplicate}`);
    }
}

if (errors.length) {
    console.error(`Site check failed with ${errors.length} error(s):`);
    errors.forEach((error) => console.error(`- ${error}`));
    process.exit(1);
}

console.log(`Site check passed: ${postSlugs.length} posts and ${htmlFiles.length} HTML files.`);
