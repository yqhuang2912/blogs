#!/usr/bin/env node

const fs = require('fs/promises');
const path = require('path');
const crypto = require('crypto');

const ROOT = path.resolve(__dirname, '..');
const OUTPUT = path.join(ROOT, '_site');
const PUBLIC_ENTRIES = [
    'index.html',
    'styles.css',
    'script.js',
    'assets',
    'icons',
    'partials',
    'posts',
];
const MATHJAX_URL = 'https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-mml-chtml.js';
const MATHJAX_INTEGRITY = 'sha384-Wuix6BuhrWbjDBs24bXrjf4ZQ5aFeFWBuKkFekO2t8xFU0iNaLQfp2K6/1Nxveei';

async function listHtmlFiles(directory) {
    const entries = await fs.readdir(directory, { withFileTypes: true });
    const files = [];
    for (const entry of entries) {
        const target = path.join(directory, entry.name);
        if (entry.isDirectory()) {
            files.push(...await listHtmlFiles(target));
        } else if (entry.name.endsWith('.html')) {
            files.push(target);
        }
    }
    return files;
}

async function optimizeHtml() {
    const files = await listHtmlFiles(OUTPUT);
    const scriptHash = crypto.createHash('sha256')
        .update(await fs.readFile(path.join(OUTPUT, 'script.js')))
        .digest('hex')
        .slice(0, 12);
    const styleHash = crypto.createHash('sha256')
        .update(await fs.readFile(path.join(OUTPUT, 'styles.css')))
        .digest('hex')
        .slice(0, 12);
    for (const file of files) {
        let html = await fs.readFile(file, 'utf8');
        html = html.replace(
            /<script src="https:\/\/cdn\.jsdelivr\.net\/npm\/mathjax@3\/es5\/tex-mml-chtml\.js" id="MathJax-script" async><\/script>/g,
            `<script src="${MATHJAX_URL}" id="MathJax-script" integrity="${MATHJAX_INTEGRITY}" crossorigin="anonymous" async></script>`,
        );
        html = html
            .replace(/(src="(?:\.\.\/)?script\.js)(?:\?[^"\s]*)?"/g, `$1?v=${scriptHash}"`)
            .replace(/(href="(?:\.\.\/)?styles\.css)(?:\?[^"\s]*)?"/g, `$1?v=${styleHash}"`);
        html = html.replace(/<img\b(?![^>]*\bloading=)([^>]*)>/gi, '<img loading="lazy" decoding="async"$1>');
        await fs.writeFile(file, html, 'utf8');
    }
}

async function main() {
    if (path.dirname(OUTPUT) !== ROOT || path.basename(OUTPUT) !== '_site') {
        throw new Error(`Unsafe build output path: ${OUTPUT}`);
    }

    await fs.rm(OUTPUT, { recursive: true, force: true });
    await fs.mkdir(OUTPUT, { recursive: true });

    for (const entry of PUBLIC_ENTRIES) {
        const source = path.join(ROOT, entry);
        const destination = path.join(OUTPUT, entry);
        await fs.cp(source, destination, { recursive: true });
    }
    await optimizeHtml();
    await fs.writeFile(path.join(OUTPUT, '.nojekyll'), '', 'utf8');
    console.log(`Built public site with ${PUBLIC_ENTRIES.length} entries in _site/.`);
}

main().catch((error) => {
    console.error(error.message);
    process.exit(1);
});
