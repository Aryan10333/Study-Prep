import os
import re
import markdown
import subprocess
from pygments.formatters import HtmlFormatter

def get_browser_path():
    """Dynamically locate Edge or Chrome browser binary across standard paths."""
    candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Google Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError("No Microsoft Edge or Google Chrome executable found on this system.")

def preprocess_markdown(text):
    """Preprocess markdown list structures and relative paths for python-markdown."""
    # Correct relative image paths from modules/ level to root level
    text = text.replace("../plots/", "plots/")

    # Double the indentation of 2-space nested lists to 4-space
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

    # Insert blank lines before list blocks if they are missing
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

def compile_document(md_files, html_out_path, pdf_out_path, page_title, header_label):
    base_dir = r"d:\Study\Prep\machine-learning-prep\generative-ai-and-agentic-ai\00_nlp_fundamentals"
    alert_types = {
        'NOTE': ('#2563eb', '#eff6ff', '#1e40af', '📌 NOTE'),
        'TIP': ('#059669', '#ecfdf5', '#065f46', '💡 TIP'),
        'IMPORTANT': ('#7c3aed', '#f5f3ff', '#5b21b6', '⚡ IMPORTANT'),
        'WARNING': ('#d97706', '#fffbeb', '#92400e', '⚠️ WARNING'),
        'CAUTION': ('#dc2626', '#fef2f2', '#991b1b', '🚨 CAUTION')
    }

    modules_html = []

    for file_name in md_files:
        full_path = os.path.join(base_dir, "modules", file_name)
        if not os.path.exists(full_path):
            print(f"ERROR: File not found: {full_path}")
            continue

        with open(full_path, "r", encoding="utf-8") as f:
            md_text = f.read()

        md_text = preprocess_markdown(md_text)

        # Preprocess custom alert tags
        for alert, (border, bg, color, label) in alert_types.items():
            pattern = re.compile(rf'>\s*\[!{alert}\]\s*\n((?:>[^\n]*\n?)*)', re.IGNORECASE)
            def alert_replacer(match):
                body_lines = match.group(1).replace('>', '').strip()
                return f'<div style="border-left: 4px solid {border}; background-color: {bg}; color: {color}; padding: 12px 16px; margin: 16px 0; border-radius: 0 6px 6px 0;"><strong>{label}:</strong><div style="margin-top: 4px;">{body_lines}</div></div>\n'
            md_text = pattern.sub(alert_replacer, md_text)

        # Protect math blocks
        math_blocks = []
        def store_math(m):
            math_blocks.append(m.group(0))
            return f"MATHPLACEHOLDER{len(math_blocks)-1}ENDMATH"

        md_text = re.sub(r'\$\$[\s\S]*?\$\$', store_math, md_text)
        md_text = re.sub(r'(?<!\$)\$[^$\n]+\$(?!\$)', store_math, md_text)

        # Convert to HTML
        html_body = markdown.markdown(
            md_text,
            extensions=['fenced_code', 'tables', 'toc', 'nl2br', 'sane_lists', 'codehilite']
        )

        # Restore math blocks
        for idx, block in enumerate(math_blocks):
            html_body = html_body.replace(f"MATHPLACEHOLDER{idx}ENDMATH", block)

        # Wrap in module container
        module_html = f"""
    <div class="module-container">
        <div style="background-color: #f8fafc; border-bottom: 2px solid #cbd5e1; padding: 12px 0; margin-bottom: 24px;">
            <span style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.1em; color: #64748b; font-weight: 600;">{header_label}</span>
        </div>
        {html_body}
    </div>
"""
        modules_html.append(module_html)

    pygments_css = HtmlFormatter(style='monokai').get_style_defs('.codehilite')
    body_content = "\n".join(modules_html)

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
        img {{
            max-width: 100\%;
            height: auto;
            display: block;
            margin: 20px auto;
            border-radius: 6px;
            border: 1px solid #e2e8f0;
        }}
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
            color: #64748b !important;
            font-size: 16px !important;
            top: -1px !important;
        }}
        .module-container ol {{
            padding-left: 20px !important;
            margin-top: 6px !important;
            margin-bottom: 12px !important;
        }}
        .module-container ol li {{
            margin-bottom: 8px !important;
            font-size: 14.5px !important;
            line-height: 1.6 !important;
            color: #334155 !important;
        }}
        table {{
            width: 100\%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 13.5px;
        }}
        th, td {{
            border: 1px solid #cbd5e1;
            padding: 8px 12px;
            text-align: left;
        }}
        th {{
            background-color: #f1f5f9;
            color: #0f172a;
            font-weight: 600;
        }}
        tr:nth-child(even) {{
            background-color: #f8fafc;
        }}
        pre, code {{
            font-family: Consolas, Monaco, 'Andale Mono', monospace;
            background-color: #f1f5f9;
            border-radius: 4px;
        }}
        code {{
            padding: 2px 4px;
            font-size: 13px;
            color: #0f172a;
        }}
        pre code {{
            padding: 0;
            font-size: 12px;
            color: inherit;
            background-color: transparent;
        }}
        {pygments_css}
        .codehilite {{
            background-color: #272822 !important;
            padding: 12px;
            border-radius: 6px;
            margin: 16px 0;
            overflow-x: auto;
        }}
        .codehilite pre {{
            margin: 0;
            background-color: transparent !important;
        }}
        .codehilite code {{
            background-color: transparent !important;
            color: #f8f8f2 !important;
        }}
        blockquote {{
            border-left: 4px solid #3b82f6;
            background-color: #eff6ff;
            margin: 12px 0;
            padding: 8px 14px;
            color: #1e40af;
            border-radius: 0 4px 4px 0;
        }}
        .katex-display {{
            margin: 14px 0 !important;
            overflow-x: auto;
            overflow-y: hidden;
        }}
        .module-container {{
            page-break-before: always;
        }}
    </style>
</head>
<body>

<div class="cover-page" style="text-align: center; padding-top: 140px; page-break-after: always;">
    <div style="display: inline-block; background: #eff6ff; color: #2563eb; padding: 6px 16px; border-radius: 20px; font-weight: 600; font-size: 13px; margin-bottom: 20px;">
        MACHINE LEARNING INTERVIEW PREPARATION
    </div>
    <h1 style="font-size: 34px; color: #0f172a; margin-bottom: 12px; line-height: 1.25; border: none; padding: 0;">
        {page_title}
    </h1>
    <h2 style="font-size: 20px; color: #3b82f6; font-weight: 500; margin-bottom: 40px; border: none;">
        Comprehensive Classical NLP Foundations & Systems
    </h2>
    <hr style="border: 0; height: 3px; background: linear-gradient(90deg, #3b82f6, #8b5cf6); width: 35%; margin: 30px auto; border-radius: 2px;" />
    
    <div style="margin-top: 60px; font-size: 14px; color: #475569; line-height: 1.8;">
        <p><b>Target Persona:</b> Senior AI Engineer / Applied AI Engineer (~3 YOE)</p>
        <p><b>Focus Areas:</b> Text Preprocessing, Vector Spaces, BM25, Embeddings, Language Models, RNN/LSTM/GRU, Drift Detection</p>
        <p><b>Includes:</b> Step-by-Step Numerical Hand Calculations, PyTorch Implementations, Production Telemetry</p>
    </div>
</div>

{body_content}

</body>
</html>
"""

    with open(html_out_path, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"Created HTML file at: {html_out_path}")

    # Generate PDF using Microsoft Edge Headless CLI
    edge_path = get_browser_path()
    temp_user_data = os.path.join(os.environ.get("TEMP", r"C:\Windows\Temp"), "edge_pdf_dir_tmp_nlp")
    
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
    
    print(f"Running browser PDF compilation for {os.path.basename(pdf_out_path)}...")
    subprocess.run(cmd, capture_output=True, text=True)
    print(f"SUCCESS: PDF generated at: {pdf_out_path}")

def main():
    base_dir = r"d:\Study\Prep\machine-learning-prep\generative-ai-and-agentic-ai\00_nlp_fundamentals"
    
    # 1. Compile Standalone Interview Cheatsheet
    print("\n--- Compiling Standalone NLP Interview Cheatsheet ---")
    cheatsheet_html = os.path.join(base_dir, "nlp_fundamentals_interview_cheatsheet.html")
    cheatsheet_pdf = os.path.join(base_dir, "nlp_fundamentals_interview_cheatsheet.pdf")
    compile_document(
        ["11_interview_questions.md"],
        cheatsheet_html,
        cheatsheet_pdf,
        "Classical NLP Foundations: Interview Questions & Answers",
        "NLP Foundations Interview Q&A"
    )

    # 2. Compile Master Study Guide
    print("\n--- Compiling Master NLP Study Guide ---")
    master_html = os.path.join(base_dir, "nlp_fundamentals_master_study_guide.html")
    master_pdf = os.path.join(base_dir, "nlp_fundamentals_master_study_guide.pdf")
    
    master_modules = [
        "01_nlp_intro_tasks.md",
        "02_text_preprocessing.md",
        "03_tokenization_subwords.md",
        "04_vector_space_models.md",
        "05_word2vec.md",
        "06_glove_fasttext.md",
        "07_statistical_language_models.md",
        "08_evaluation_metrics.md",
        "09_rnn_lstm_gru.md",
        "10_production_nlp_pipelines.md",
        "11_interview_questions.md"
    ]
    compile_document(
        master_modules,
        master_html,
        master_pdf,
        "Classical NLP Foundations: Master Study Guide",
        "NLP Foundations Curriculum Guide"
    )

if __name__ == "__main__":
    main()
