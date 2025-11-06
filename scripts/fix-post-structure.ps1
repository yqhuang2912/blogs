# 修复 post 文件结构：将 sidebar 移到 main-content 内部

$postFiles = Get-ChildItem -Path "posts" -Filter "*.html" | Where-Object { $_.Name -ne "11395.html" -and $_.Name -ne "manifest.json" }

foreach ($file in $postFiles) {
    Write-Host "Processing: $($file.Name)"
    $content = Get-Content $file.FullName -Raw -Encoding UTF8
    
    # 检查是否已经是新结构
    if ($content -match '<main class="main-content">\s+<div class="wrapper">') {
        Write-Host "  Already updated, skipping..."
        continue
    }
    
    # 替换：移除外层 wrapper，将 main 提升
    $content = $content -replace '    <div class="wrapper">\s+        <main class="main-content">', '    <main class="main-content">' + "`n" + '        <div class="wrapper">'
    
    # 替换：移除 </main>，在 article 结束后关闭 wrapper
    $content = $content -replace '</article>\s+        </main>\s+        <div data-include="\.\./partials/sidebar-post\.html"></div>\s+    </div>', '</article>' + "`n" + '        </div>' + "`n`n" + '        <div data-include="../partials/sidebar-post.html"></div>' + "`n" + '    </main>'
    
    Set-Content -Path $file.FullName -Value $content -Encoding UTF8 -NoNewline
    Write-Host "  Updated successfully"
}

Write-Host "`nAll files processed!"
