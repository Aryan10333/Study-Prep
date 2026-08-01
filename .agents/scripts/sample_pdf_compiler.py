import os
import re
import shutil
import markdown
import tempfile
import subprocess
from pygments.formatters import HtmlFormatter

def get_browser_path():
    """Dynamically locate Edge or Chrome browser binary across standard paths."""
    candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Google Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        shutil.which("msedge"),
        shutil.which("chrome"),
        shutil.which("google-chrome"),
        shutil.which("chromium"),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError("No Microsoft Edge or Google Chrome executable found on this system.")

def preprocess_markdown(text):
    """Preprocess markdown list structures and spacings for python-markdown."""
    # 1. Double the indentation of 2-space nested lists to 4-space
    lines = text.split('\n')
    new_lines = []
    for line in lines:
        m = re.match(r'^( +)([-*]|\d+\.) (.*)', line)
        if m:
            spaces, marker, content = m.groups()
            if len(spaces) == 2:
                line = '    ' + marker + ' ' + content
        new_lines.append(line)
    text = '\n'.join(new_lines)

    # 2. Insert blank lines before list blocks if they are missing
    lines = text.split('\n')
    new_lines = []
    for i, line in enumerate(lines):
        if i > 0:
            stripped = line.lstrip()
            prev_stripped = lines[i-1].lstrip()
            is_list_start = stripped.startswith('- ') or stripped.startswith('* ') or re.match(r'^\d+\.\s', stripped)
            if is_list_start:
                if prev_stripped != "" and not prev_stripped.startswith('- ') and not prev_stripped.startswith('* ') and not re.match(r'^\d+\.\s', prev_stripped) and not prev_stripped.startswith('#') and not prev_stripped.startswith('>'):
                    new_lines.append("")
        new_lines.append(line)
    return '\n'.join(new_lines)

def compile_markdown_to_pdf(md_paths, html_out_path, pdf_out_path, page_title="Master Study Guide"):
    """Compiles single/multiple markdown source files into styled HTML and printable PDF."""
    alert_types = {
        'NOTE': ('#2563eb', '#eff6ff', '#1e40af', '📌 NOTE'),
        'TIP': ('#059669', '#ecfdf5', '#065f46', '💡 TIP'),
        'IMPORTANT': ('#7c3aed', '#f5f3ff', '#5b21b6', '⚡ IMPORTANT'),
        'WARNING': ('#d97706', '#fffbeb', '#92400e', '⚠️ WARNING'),
        'CAUTION': ('#dc2626', '#fef2f2', '#991b1b', '🚨 CAUTION')
    }

    modules_html = []

    for md_path in md_paths:
        if not os.path.exists(md_path):
            print(f"WARNING: File not found: {md_path}")
            continue

        with open(md_path, "r", encoding="utf-8") as f:
            md_text = f.read()

        # Step 1: Preprocess lists
        md_text = preprocess_markdown(md_text)

        # Step 2: Parse and replace custom alerts
        for alert, (border, bg, color, label) in alert_types.items():
            pattern = re.compile(rf'>\s*\[!{alert}\]\s*\n((?:>[^\n]*\n?)*)', re.IGNORECASE)
            def alert_replacer(match):
                lines = match.group(1).split('\n')
                cleaned_lines = []
                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith('>'):
                        cleaned_lines.append(stripped[1:].strip())
                    else:
                        cleaned_lines.append(stripped)
                body_markdown = '\n'.join(cleaned_lines).strip()
                body_html = markdown.markdown(body_markdown, extensions=['fenced_code', 'tables', 'nl2br', 'sane_lists', 'codehilite'])
                return f'<div style="border-left: 4px solid {border}; background-color: {bg}; color: {color}; padding: 12px 16px; margin: 16px 0; border-radius: 0 6px 6px 0;"><strong>{label}:</strong><div style="margin-top: 4px;">{body_html}</div></div>\n'
            md_text = pattern.sub(alert_replacer, md_text)

        # Step 3: Extract and safeguard KaTeX math blocks
        math_blocks = []
        def store_math(m):
            math_blocks.append(m.group(0))
            return f"MATHPLACEHOLDER{len(math_blocks)-1}ENDMATH"

        md_text = re.sub(r'\$\$[\s\S]*?\$\$', store_math, md_text)
        md_text = re.sub(r'(?<!\$)\$[^$\n]+\$(?!\$)', store_math, md_text)

        # Step 4: Markdown parsing
        html_body = markdown.markdown(
            md_text,
            extensions=['fenced_code', 'tables', 'toc', 'nl2br', 'sane_lists', 'codehilite']
        )

        # Step 5: Restore math blocks and escape brackets inside math for headless browser compatibility
        for idx, block in enumerate(math_blocks):
            escaped_block = block.replace('<', '&lt;').replace('>', '&gt;')
            html_body = html_body.replace(f"MATHPLACEHOLDER{idx}ENDMATH", escaped_block)

        # Wrap in module container
        module_html = f"""
        <div class="module-container">
            {html_body}
        </div>
        """
        modules_html.append(module_html)

    # Pygments syntax highlighter styling (Monokai dark theme)
    pygments_css = HtmlFormatter(style='monokai').get_style_defs('.codehilite')
    body_content = "\n".join(modules_html)

    # Premium A4 HTML Template
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>{page_title}</title>
    <!-- Load KaTeX -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"
            onload="renderMathInElement(document.body, {{
                delimiters: [
                    {{left: '$$', right: '$$', display: true}},
                    {{left: '$', right: '$', display: false}}
                ]
            }});"></script>
    <style>
        @page {{
            size: A4;
            margin: 20mm 18mm 20mm 18mm;
        }}
        body {{
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
            font-size: 14.5px;
            line-height: 1.6;
            color: #334155;
            background-color: #ffffff;
            margin: 0;
            padding: 0;
        }}
        h1 {{
            font-size: 22px;
            color: #0f172a;
            border-bottom: 2px solid #3b82f6;
            padding-bottom: 6px;
            margin-top: 30px;
            margin-bottom: 16px;
            page-break-after: avoid;
            break-after: avoid;
        }}
        h2 {{
            font-size: 18px;
            color: #2563eb;
            margin-top: 24px;
            margin-bottom: 12px;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 4px;
            page-break-after: avoid;
            break-after: avoid;
        }}
        h3 {{
            font-size: 15px;
            color: #0284c7;
            margin-top: 20px;
            margin-bottom: 10px;
            page-break-after: avoid;
            break-after: avoid;
        }}
        h4, h5, h6 {{
            font-size: 14.5px !important;
            font-weight: 700 !important;
            color: #0f172a !important;
            margin-top: 18px !important;
            margin-bottom: 8px !important;
            page-break-after: avoid;
            break-after: avoid;
        }}
        
        /* Hide scrollbars globally in PDF print */
        ::-webkit-scrollbar {{
            display: none !important;
            width: 0 !important;
            height: 0 !important;
            background: transparent !important;
        }}
        
        a {{
            color: #2563eb;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        
        p {{
            font-size: 14.5px !important;
            line-height: 1.6 !important;
            color: #334155;
            margin-bottom: 10px;
        }}
        
        /* Custom Bullet List and Ordered List Styling */
        .module-container ul {{
            list-style: none !important;
            padding-left: 20px !important;
            margin-top: 6px !important;
            margin-bottom: 12px !important;
        }}
        .module-container ul li {{
            position: relative !important;
            margin-bottom: 8px !important;
            padding-left: 10px !important;
            font-size: 14.5px !important;
            line-height: 1.6 !important;
            color: #334155 !important;
        }}
        .module-container ul li::before {{
            content: "•" !important;
            color: #3b82f6 !important;
            font-weight: bold !important;
            font-size: 20px !important;
            display: inline-block !important;
            width: 1em !important;
            margin-left: -1em !important;
            position: absolute !important;
            top: -3px !important;
            left: 5px !important;
        }}
        .module-container ul ul {{
            margin-top: 4px !important;
            margin-bottom: 4px !important;
        }}
        .module-container ul ul li::before {{
            content: "◦" !important;
            color: #8b5cf6 !important;
            font-size: 16px !important;
            top: -1px !important;
        }}
        
        code {{
            font-family: 'Consolas', 'Cascadia Code', monospace;
            background-color: #f1f5f9;
            color: #0f172a;
            padding: 2px 5px;
            border-radius: 4px;
            font-size: 13.5px !important;
        }}
        
        img {{
            max-width: 90% !important;
            height: auto !important;
            display: block;
            margin: 24px auto;
            border-radius: 6px;
            page-break-inside: avoid;
            break-inside: avoid;
        }}
        
        table {{
            width: 100% !important;
            border-collapse: collapse !important;
            margin: 20px 0 !important;
            font-size: 13px !important;
            border: 1.5px solid #64748b !important;
            page-break-inside: avoid;
            break-inside: avoid;
            table-layout: auto !important;
            max-width: 100% !important;
        }}
        th {{
            background-color: #0f172a !important;
            color: #ffffff !important;
            font-weight: 700 !important;
            border: 1.5px solid #475569 !important;
            padding: 10px 12px !important;
        }}
        td {{
            border: 1px solid #cbd5e1 !important;
            padding: 8px 12px !important;
            vertical-align: top !important;
        }}
        tr:nth-child(even) td {{
            background-color: #f8fafc !important;
        }}
        
        /* Pygments Contrast Overrides */
        {pygments_css}
        .codehilite, .codehilite pre, .codehilite code, .codehilite span {{
            background-color: transparent !important;
            color: #f8fafc !important;
        }}
        .codehilite {{
            background-color: #0f172a !important;
            border-radius: 6px;
            padding: 12px 16px !important;
            margin: 16px 0 !important;
            border: 1px solid #334155 !important;
            overflow-x: hidden !important;
            page-break-inside: avoid;
            break-inside: avoid;
        }}
        .codehilite pre {{
            white-space: pre-wrap !important;
            word-wrap: break-word !important;
            font-size: 12px !important;
            margin: 0 !important;
        }}
        
        .page-break {{
            page-break-before: always;
            break-before: page;
        }}
    </style>
</head>
<body>
    {body_content}
</body>
</html>
"""

    with open(html_out_path, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"HTML output generated successfully: {html_out_path}")

    # Generate PDF via headless browser printing
    browser_executable = get_browser_path()
    with tempfile.TemporaryDirectory() as temp_user_dir:
        cmd = [
            browser_executable,
            f"--user-data-dir={temp_user_dir}",
            "--headless",
            "--disable-gpu",
            "--run-all-compositor-stages-before-draw",
            "--virtual-time-budget=8000",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_out_path}",
            html_out_path
        ]
        print(f"Printing to PDF via headless browser ({browser_executable})...")
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    
    print(f"PDF generated successfully: {pdf_out_path}")

if __name__ == "__main__":
    # Self-test block using temp directories
    with tempfile.TemporaryDirectory() as test_dir:
        test_md = os.path.join(test_dir, "test.md")
        test_html = os.path.join(test_dir, "test.html")
        test_pdf = os.path.join(test_dir, "test.pdf")

        dummy_content = """# Title Module
Hello this is standard text.

> [!NOTE]
> This is a sample alert block showing information.

Here is math:
$$e^{i\\pi} + 1 = 0$$
"""
        with open(test_md, "w", encoding="utf-8") as f:
            f.write(dummy_content)

        compile_markdown_to_pdf([test_md], test_html, test_pdf, "Test Compilation")
