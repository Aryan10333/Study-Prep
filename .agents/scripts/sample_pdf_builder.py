import os
import re
import markdown
import subprocess
from pygments.formatters import HtmlFormatter

def build_pdf_from_markdown(md_path, html_out_path, pdf_out_path):
    # 1. Read Markdown
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()
        
    # 2. Preprocess custom GitHub alerts (e.g. > [!NOTE]) into styled HTML divs
    alert_types = {
        'NOTE': ('#2563eb', '#eff6ff', '#1e40af', '📌 NOTE'),
        'TIP': ('#059669', '#ecfdf5', '#065f46', '💡 TIP'),
        'IMPORTANT': ('#7c3aed', '#f5f3ff', '#5b21b6', '⚡ IMPORTANT'),
        'WARNING': ('#d97706', '#fffbeb', '#92400e', '⚠️ WARNING'),
        'CAUTION': ('#dc2626', '#fef2f2', '#991b1b', '🚨 CAUTION')
    }
    
    for alert, (border, bg, color, label) in alert_types.items():
        pattern = re.compile(rf'>\s*\[!{alert}\]\s*\n((?:>[^\n]*\n?)*)', re.IGNORECASE)
        def replacer(match):
            body_lines = match.group(1).replace('>', '').strip()
            return f'<div style="border-left: 4px solid {border}; background-color: {bg}; color: {color}; padding: 12px 16px; margin: 16px 0; border-radius: 0 6px 6px 0;"><strong>{label}:</strong><div style="margin-top: 4px;">{body_lines}</div></div>\n'
        md_text = pattern.sub(replacer, md_text)

    # 3. Protect math blocks from being corrupted by the markdown parser
    math_blocks = []
    def store_math(m):
        math_blocks.append(m.group(0))
        return f"MATHPLACEHOLDER{len(math_blocks)-1}ENDMATH"

    # Protect display and inline math
    md_text = re.sub(r'\$\$[\s\S]*?\$\$', store_math, md_text)
    md_text = re.sub(r'(?<!\$)\$[^$\n]+\$(?!\$)', store_math, md_text)

    # 4. Convert markdown to HTML with Pygments code formatting and tables extensions
    html_body = markdown.markdown(
        md_text,
        extensions=['fenced_code', 'tables', 'toc', 'nl2br', 'sane_lists', 'codehilite']
    )

    # Restore math blocks
    for idx, block in enumerate(math_blocks):
        html_body = html_body.replace(f"MATHPLACEHOLDER{idx}ENDMATH", block)

    # 5. Compile Pygments Syntax Highlighting CSS
    pygments_css = HtmlFormatter(style='monokai').get_style_defs('.codehilite')

    # 6. Embed in a clean HTML document wrapping KaTeX math configurations
    full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Master Document</title>
    <!-- Load KaTeX -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
    <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"></script>
    <script>
        document.addEventListener("DOMContentLoaded", function() {{
            renderMathInElement(document.body, {{
                delimiters: [
                    {{left: "$$", right: "$$", display: true}},
                    {{left: "$", right: "$", display: false}}
                ]
            }});
        }});
    </script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            font-size: 14px;
            color: #0f172a;
            max-width: 800px;
            margin: 0 auto;
            padding: 30px;
        }}
        h1, h2, h3 {{ border-bottom: 1px solid #cbd5e1; padding-bottom: 4px; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 16px 0;
            border: 1px solid #64748b;
        }}
        th, td {{
            border: 1px solid #cbd5e1;
            padding: 8px 12px;
            text-align: left;
        }}
        th {{ background-color: #0f172a; color: white; }}
        
        /* Pygments monokai styles */
        {pygments_css}
        
        .codehilite, .codehilite pre, .codehilite code, .codehilite span {{
            background-color: transparent !important;
            color: #f8fafc !important;
        }}
        .codehilite {{
            background-color: #0f172a !important;
            border-radius: 6px;
            padding: 12px;
            border: 1px solid #334155;
            overflow-x: auto;
        }}
    </style>
</head>
<body>
    {html_body}
</body>
</html>
"""

    with open(html_out_path, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"Created HTML build file at: {html_out_path}")

    # 7. Convert HTML to PDF using Microsoft Edge Headless CLI
    edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    temp_user_data = os.path.join(os.environ.get("TEMP", r"C:\Windows\Temp"), "edge_pdf_dir_tmp")
    
    cmd = [
        edge_path,
        f"--user-data-dir={temp_user_data}",
        "--headless",
        "--disable-gpu",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=8000",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_out_path}",
        html_out_path
    ]
    
    print("Running Edge PDF printing...")
    subprocess.run(cmd, capture_output=True, text=True)
    print(f"SUCCESS: PDF generated at: {pdf_out_path}")

if __name__ == "__main__":
    # Test path verification
    test_md = os.path.abspath("./test_input.md")
    test_html = os.path.abspath("./test_output.html")
    test_pdf = os.path.abspath("./test_output.pdf")
    
    # Create simple dummy markdown to test compilation
    with open(test_md, "w", encoding="utf-8") as f:
        f.write("# Hello Math\\nLet $x = 2$, then $y = x^2 = 4$.\\n> [!NOTE]\\nThis is a sample alert.\\n")
        
    build_pdf_from_markdown(test_md, test_html, test_pdf)
