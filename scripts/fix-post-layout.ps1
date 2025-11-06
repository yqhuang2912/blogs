# 批量修改所有 post 文件的布局结构

$postsDir = "posts"
$files = Get-ChildItem -Path $postsDir -Filter "*.html"

foreach ($file in $files) {
    $content = Get-Content -Path $file.FullName -Raw -Encoding UTF8
    
    # 找到 </main> 和 sidebar 之间的部分，将 sidebar 移到 wrapper 内
    # 查找模式: </main>\n\n        <div data-include="../partials/sidebar-post.html"></div>\n    </div>\n\n    <div data-include="../partials/footer.html"></div>
    
    $oldPattern = '(?s)</main>\s+<div data-include="../partials/sidebar-post.html"></div>\s+</div>\s+<div data-include="../partials/footer.html"></div>'
    $newPattern = '</main>' + "`r`n`r`n" + '        <div data-include="../partials/sidebar-post.html"></div>' + "`r`n" + '    </div>' + "`r`n`r`n" + '    <div data-include="../partials/footer.html"></div>'
    
    if ($content -match $oldPattern) {
        $newContent = $content -replace $oldPattern, $newPattern
        Set-Content -Path $file.FullName -Value $newContent -Encoding UTF8 -NoNewline
        Write-Host "Processed: $($file.Name)"
    } else {
        Write-Host "Skipped (pattern not found): $($file.Name)"
    }
}

Write-Host "Done!"
