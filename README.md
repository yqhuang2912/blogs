## 文章管理

推荐使用交互式管理器，不需要记忆参数：

```powershell
npm run post
```

菜单支持创建新草稿、发布已有草稿、修改、删除、查看文章、重新生成清单和站点检查。

创建新草稿时会询问标题、slug、日期、分类和标签，并生成 `drafts/<slug>.md`。新草稿默认包含 `draft: true`，不会被误发布。正文完成后，将它改成 `draft: false`，再次运行 `npm run post` 并选择“发布已有草稿”。

修改和删除可以直接从文章列表选择；删除操作需要输入文章 slug 二次确认。

其他快捷命令：

```powershell
npm run posts:list
npm run post -- --help
npm run check
```

原有命令仍可用于自动化：

```powershell
node scripts/create-post-from-markdown.js drafts/sample-post.md


node scripts/create-post-from-markdown.js drafts/sample-post.md --slug sample-post --id 12345 --title "Sample Post" --mode create

node scripts/create-post-from-markdown.js drafts/sample-post.md --mode update --title "Updated Sample Post"

node scripts/create-post-from-markdown.js --mode delete --slug sample-post --id 12345
```
