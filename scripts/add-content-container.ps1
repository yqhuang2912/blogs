# 重构 post 文件：移除 content-container，将 sidebar 放到 main-content 中

$postFiles = Get-ChildItem -Path "posts" -Filter "*.html" -Exclude "11395.html"

foreach ($file in $postFiles) {
    $content = Get-Content $file.FullName -Raw -Encoding UTF8
    
    # 移除 content-container，将其改为 main
    $content = $content -replace '    <div class="content-container">\s+        <div class="wrapper">\s+            <main class="main-content">', '    <main class="main-content">' + "`n" + '        <div class="wrapper">'
    
    # 将 </main> 移到 sidebar 之后
    $content = $content -replace '                </article>\s+            </main>\s+            <div data-include="\.\./partials/sidebar-post\.html"></div>\s+        </div>\s+    </div>', '                </article>' + "`n" + '        </div>' + "`n`n" + '        <div data-include="../partials/sidebar-post.html"></div>' + "`n" + '    </main>'
    
    Set-Content -Path $file.FullName -Value $content -Encoding UTF8 -NoNewline
    Write-Host "Updated: $($file.Name)"
}

Write-Host "`nAll post files updated successfully!"
