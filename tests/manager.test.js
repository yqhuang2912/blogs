const assert = require('node:assert/strict');
const test = require('node:test');
const {
    buildCreateArgs,
    buildDeleteArgs,
    buildDraftTemplate,
    buildUpdateArgs,
    normalizeDraftSlug,
    parseCommaList,
} = require('../scripts/manage-posts');

test('create arguments omit empty optional values', () => {
    assert.deepEqual(buildCreateArgs('drafts/example.md', {}), [
        'drafts/example.md', '--mode', 'create',
    ]);
    assert.deepEqual(buildCreateArgs('drafts/example.md', {
        slug: 'example', id: '12000', title: '示例',
    }), [
        'drafts/example.md', '--mode', 'create',
        '--slug', 'example', '--id', '12000', '--title', '示例',
    ]);
});

test('update arguments preserve the selected post identity by default', () => {
    const post = { id: '11406', slug: 'diffusion-ddpm' };
    assert.deepEqual(buildUpdateArgs('drafts/diffusion-ddpm.md', post), [
        'drafts/diffusion-ddpm.md', '--mode', 'update',
        '--id', '11406', '--slug', 'diffusion-ddpm',
    ]);
});

test('delete arguments always include both slug and id', () => {
    assert.deepEqual(buildDeleteArgs({ id: '11406', slug: 'diffusion-ddpm' }), [
        '--mode', 'delete', '--slug', 'diffusion-ddpm', '--id', '11406',
    ]);
});

test('draft slugs reject paths and normalize safe input', () => {
    assert.equal(normalizeDraftSlug(' New-Post_2 '), 'new-post_2');
    assert.throws(() => normalizeDraftSlug('../outside'), /slug/i);
    assert.throws(() => normalizeDraftSlug('中文标题'), /slug/i);
});

test('draft templates are unpublished and preserve taxonomy values', () => {
    const markdown = buildDraftTemplate({
        title: '新的文章',
        slug: 'new-post',
        createdAt: '2026-08-30',
        categories: ['人工智能'],
        tags: ['生成模型', '数学'],
    });
    assert.match(markdown, /title: "新的文章"/);
    assert.match(markdown, /slug: new-post/);
    assert.match(markdown, /draft: true/);
    assert.match(markdown, /  - "生成模型"/);
});

test('Chinese and English commas both split taxonomy input', () => {
    assert.deepEqual(parseCommaList('人工智能，数学, 生成模型'), ['人工智能', '数学', '生成模型']);
});
