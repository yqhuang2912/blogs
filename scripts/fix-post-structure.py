import os
import re
import glob

# 获取所有 post 文件，排除 11395.html（已经手动更新）
post_files = [f for f in glob.glob("posts/*.html") if os.path.basename(f) != "11395.html" and os.path.basename(f) != "manifest.json"]

for filepath in post_files:
    print(f"Processing: {os.path.basename(filepath)}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否已经是新结构
    if '<main class="main-content">\n        <div class="wrapper">' in content:
        print(f"  Already updated, skipping...")
        continue
    
    # 1. 替换开始标签：从 wrapper > main 改为 main > wrapper
    content = re.sub(
        r'    <div class="wrapper">\s+        <main class="main-content">',
        '    <main class="main-content">\n        <div class="wrapper">',
        content
    )
    
    # 2. 移除重复的 </article> 标签
    content = re.sub(r'</article>\s+</article>', '</article>', content)
    
    # 3. 替换结束标签：关闭 wrapper，然后 sidebar，然后关闭 main
    content = re.sub(
        r'            </article>\s+</main>\s+        <div data-include="\.\./partials/sidebar-post\.html"></div>\s+    </div>',
        '            </article>\n        </div>\n\n        <div data-include="../partials/sidebar-post.html"></div>\n    </main>',
        content
    )
    
    with open(filepath, 'w', encoding='utf-8', newline='') as f:
        f.write(content)
    
    print(f"  Updated successfully")

print("\nAll files processed!")
