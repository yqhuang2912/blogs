#!/usr/bin/env node

const fs = require('fs/promises');
const path = require('path');
const { spawn } = require('child_process');
const { createInterface } = require('readline/promises');
const { stdin, stdout } = require('process');

const ROOT = path.resolve(__dirname, '..');
const DRAFTS_DIR = path.join(ROOT, 'drafts');
const MANIFEST_PATH = path.join(ROOT, 'posts', 'manifest.json');
const POST_CLI = path.join(ROOT, 'scripts', 'create-post-from-markdown.js');
const MANIFEST_CLI = path.join(ROOT, 'scripts', 'generate-post-manifest.js');
const CHECK_CLI = path.join(ROOT, 'scripts', 'check-site.js');

function printHelp() {
    console.log(`文章管理器

用法：
  npm run post            打开交互菜单
  npm run post -- --list  列出现有文章
  npm run post -- --help  显示帮助

交互菜单支持：创建草稿、发布草稿、修改、删除、生成清单和站点检查。`);
}

async function loadPosts() {
    const raw = await fs.readFile(MANIFEST_PATH, 'utf8');
    const manifest = JSON.parse(raw);
    return Array.isArray(manifest.posts) ? manifest.posts : [];
}

function printPosts(posts) {
    if (!posts.length) {
        console.log('暂无文章。');
        return;
    }
    console.table(posts.map((post, index) => ({
        序号: index + 1,
        ID: post.id,
        slug: post.slug,
        标题: post.title,
        日期: post.createdAt || '',
    })));
}

async function listMarkdownFiles(directory = DRAFTS_DIR) {
    const entries = await fs.readdir(directory, { withFileTypes: true });
    const files = [];
    for (const entry of entries) {
        const target = path.join(directory, entry.name);
        if (entry.isDirectory()) {
            if (entry.name !== 'images') {
                files.push(...await listMarkdownFiles(target));
            }
        } else if (entry.name.endsWith('.md')) {
            files.push(target);
        }
    }
    return files.sort((a, b) => a.localeCompare(b, 'zh-Hans'));
}

async function askChoice(rl, question, items) {
    if (!items.length) {
        throw new Error('没有可供选择的项目。');
    }
    items.forEach((item, index) => console.log(`${index + 1}. ${item.label}`));
    while (true) {
        const answer = (await rl.question(`${question}（1-${items.length}，输入 0 取消）：`)).trim();
        if (answer === '0') {
            return null;
        }
        const index = Number.parseInt(answer, 10) - 1;
        if (Number.isInteger(index) && index >= 0 && index < items.length) {
            return items[index].value;
        }
        console.log('请输入列表中的有效序号。');
    }
}

function appendOption(args, flag, value) {
    const normalized = String(value ?? '').trim();
    if (normalized) {
        args.push(flag, normalized);
    }
}

function normalizeDraftSlug(value) {
    const slug = String(value ?? '').trim().toLowerCase();
    if (!/^[a-z0-9]+(?:[-_][a-z0-9]+)*$/.test(slug)) {
        throw new Error('slug 只能包含小写字母、数字、连字符和下划线。');
    }
    return slug;
}

function todayLocal() {
    const now = new Date();
    const year = now.getFullYear();
    const month = String(now.getMonth() + 1).padStart(2, '0');
    const day = String(now.getDate()).padStart(2, '0');
    return `${year}-${month}-${day}`;
}

function parseCommaList(value) {
    return String(value ?? '')
        .split(/[，,]/)
        .map((item) => item.trim())
        .filter(Boolean);
}

function buildDraftTemplate({ title, slug, createdAt, categories = [], tags = [] }) {
    const yamlField = (name, items) => items.length
        ? `${name}:\n${items.map((item) => `  - ${JSON.stringify(item)}`).join('\n')}`
        : `${name}: []`;
    return `---
title: ${JSON.stringify(title)}
slug: ${slug}
createdAt: ${createdAt}
${yamlField('categories', categories)}
${yamlField('tags', tags)}
draft: true
---

在这里开始编写正文。
`;
}

function buildCreateArgs(source, { slug, id, title } = {}) {
    const args = [source, '--mode', 'create'];
    appendOption(args, '--slug', slug);
    appendOption(args, '--id', id);
    appendOption(args, '--title', title);
    return args;
}

function buildUpdateArgs(source, post, { slug, title } = {}) {
    const args = [source, '--mode', 'update', '--id', String(post.id), '--slug', slug || post.slug];
    appendOption(args, '--title', title);
    return args;
}

function buildDeleteArgs(post) {
    return ['--mode', 'delete', '--slug', post.slug, '--id', String(post.id)];
}

async function runNode(script, args = []) {
    await new Promise((resolve, reject) => {
        const child = spawn(process.execPath, [script, ...args], {
            cwd: ROOT,
            stdio: 'inherit',
        });
        child.once('error', reject);
        child.once('close', (code) => {
            if (code === 0) {
                resolve();
            } else {
                reject(new Error(`命令执行失败，退出码：${code}`));
            }
        });
    });
}

async function runPostCommand(args) {
    await runNode(POST_CLI, args);
    console.log('\n正在检查站点完整性…');
    await runNode(CHECK_CLI);
    console.log('✅ 文章操作和站点检查均已完成。');
}

async function chooseDraft(rl, preferredSlug = '') {
    const files = await listMarkdownFiles();
    const preferred = preferredSlug
        ? files.find((file) => path.basename(file, '.md') === preferredSlug)
        : null;
    if (preferred) {
        const usePreferred = (await rl.question(`找到源稿 drafts/${path.basename(preferred)}，使用它？(Y/n)：`)).trim().toLowerCase();
        if (!usePreferred || usePreferred === 'y' || usePreferred === 'yes') {
            return preferred;
        }
    }
    return askChoice(rl, '请选择 Markdown 源稿', files.map((file) => ({
        label: path.relative(ROOT, file).replace(/\\/g, '/'),
        value: file,
    })));
}

async function createDraft(rl) {
    const title = (await rl.question('文章标题：')).trim();
    if (!title) {
        throw new Error('文章标题不能为空。');
    }

    let slug;
    while (!slug) {
        const answer = await rl.question('slug（小写英文、数字、- 或 _）：');
        try {
            slug = normalizeDraftSlug(answer);
        } catch (error) {
            console.log(error.message);
        }
    }

    const target = path.join(DRAFTS_DIR, `${slug}.md`);
    try {
        await fs.access(target);
        throw new Error(`草稿已存在：drafts/${slug}.md`);
    } catch (error) {
        if (error.code !== 'ENOENT') throw error;
    }

    const defaultDate = todayLocal();
    const createdAt = (await rl.question(`日期（默认 ${defaultDate}）：`)).trim() || defaultDate;
    if (Number.isNaN(new Date(createdAt).getTime())) {
        throw new Error(`日期格式无效：${createdAt}`);
    }
    const categories = parseCommaList(await rl.question('分类（多个用逗号分隔，可留空）：'));
    const tags = parseCommaList(await rl.question('标签（多个用逗号分隔，可留空）：'));
    const template = buildDraftTemplate({ title, slug, createdAt, categories, tags });
    await fs.writeFile(target, template, { encoding: 'utf8', flag: 'wx' });

    console.log(`\n✅ 已创建 drafts/${slug}.md`);
    console.log('请编辑正文；完成后把 front matter 中的 draft: true 改为 draft: false，再运行 npm run post 发布。');
}

async function createPost(rl) {
    const source = await chooseDraft(rl);
    if (!source) return;
    const slug = await rl.question('slug（留空则自动推导）：');
    const id = await rl.question('文章 ID（留空则自动生成）：');
    const title = await rl.question('覆盖标题（留空使用 front matter）：');
    await runPostCommand(buildCreateArgs(path.relative(ROOT, source), { slug, id, title }));
}

async function updatePost(rl, posts) {
    const post = await askChoice(rl, '请选择要修改的文章', posts.map((item) => ({
        label: `[${item.id}] ${item.title} (${item.slug})`,
        value: item,
    })));
    if (!post) return;
    const source = await chooseDraft(rl, post.slug);
    if (!source) return;
    const slug = await rl.question(`slug（当前 ${post.slug}，留空保持）：`);
    const title = await rl.question('覆盖标题（留空使用 front matter）：');
    await runPostCommand(buildUpdateArgs(path.relative(ROOT, source), post, { slug, title }));
}

async function deletePost(rl, posts) {
    const post = await askChoice(rl, '请选择要删除的文章', posts.map((item) => ({
        label: `[${item.id}] ${item.title} (${item.slug})`,
        value: item,
    })));
    if (!post) return;
    console.log('\n将删除：');
    console.log(`- posts/${post.slug}.html`);
    console.log(`- assets/${post.id}/（如果存在）`);
    console.log('- 对应 manifest 条目将在删除后重新生成');
    const confirmation = (await rl.question(`请输入文章 slug “${post.slug}” 确认删除：`)).trim();
    if (confirmation !== post.slug) {
        console.log('确认内容不匹配，已取消删除。');
        return;
    }
    await runPostCommand(buildDeleteArgs(post));
}

async function interactiveMain() {
    const rl = createInterface({ input: stdin, output: stdout });
    try {
        const action = await askChoice(rl, '\n请选择操作', [
            { label: '创建新草稿', value: 'draft' },
            { label: '发布已有草稿', value: 'create' },
            { label: '修改文章', value: 'update' },
            { label: '删除文章', value: 'delete' },
            { label: '查看现有文章', value: 'list' },
            { label: '重新生成文章清单', value: 'manifest' },
            { label: '运行站点检查', value: 'check' },
        ]);
        if (!action) return;
        const posts = ['update', 'delete', 'list'].includes(action) ? await loadPosts() : [];
        if (action === 'draft') await createDraft(rl);
        if (action === 'create') await createPost(rl);
        if (action === 'update') await updatePost(rl, posts);
        if (action === 'delete') await deletePost(rl, posts);
        if (action === 'list') printPosts(posts);
        if (action === 'manifest') await runNode(MANIFEST_CLI);
        if (action === 'check') await runNode(CHECK_CLI);
    } finally {
        rl.close();
    }
}

async function main() {
    const option = process.argv[2];
    if (option === '--help' || option === '-h') {
        printHelp();
        return;
    }
    if (option === '--list' || option === '-l') {
        printPosts(await loadPosts());
        return;
    }
    if (option) {
        throw new Error(`未知参数：${option}。使用 --help 查看帮助。`);
    }
    await interactiveMain();
}

module.exports = {
    buildCreateArgs,
    buildDeleteArgs,
    buildDraftTemplate,
    buildUpdateArgs,
    normalizeDraftSlug,
    parseCommaList,
};

if (require.main === module) {
    main().catch((error) => {
        console.error(`❌ ${error.message}`);
        process.exitCode = 1;
    });
}
