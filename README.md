```nodejs
node scripts/create-post-from-markdown.js drafts/sample-post.md


node scripts/create-post-from-markdown.js drafts/sample-post.md --slug sample-post --id 12345 --title "Sample Post" --mode create

node scripts/create-post-from-markdown.js drafts/sample-post.md --mode update --title "Updated Sample Post"

node scripts/create-post-from-markdown.js --mode delete --slug sample-post --id 12345
```