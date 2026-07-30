import os
import re
import markdown
import subprocess
from pygments.formatters import HtmlFormatter

def compile_master_guide():
    base_dir = r"d:\Study\Prep\machine-learning-prep\generative-ai-and-agentic-ai\00_nlp_fundamentals"
    html_out_path = os.path.join(base_dir, "nlp_master_study_guide.html")
    pdf_out_path = os.path.join(base_dir, "nlp_master_study_guide.pdf")

    # Define the 10 modules in order
    module_files = [
        "01_nlp_introduction.md",
        "02_text_preprocessing.md",
        "03_text_representation.md",
        "04_statistical_language_models.md",
        "05_word_embeddings.md",
        "06_sequence_models.md",
        "07_attention_and_transformer_prerequisites.md",
        "08_nlp_evaluation.md",
        "09_production_nlp.md",
        "10_interview_questions.md"
    ]

    alert_types = {
        'NOTE': ('#2563eb', '#eff6ff', '#1e40af', '📌 NOTE'),
        'TIP': ('#059669', '#ecfdf5', '#065f46', '💡 TIP'),
        'IMPORTANT': ('#7c3aed', '#f5f3ff', '#5b21b6', '⚡ IMPORTANT'),
        'WARNING': ('#d97706', '#fffbeb', '#92400e', '⚠️ WARNING'),
        'CAUTION': ('#dc2626', '#fef2f2', '#991b1b', '🚨 CAUTION')
    }

    modules_html = []

    for file_name in module_files:
        full_path = os.path.join(base_dir, "modules", file_name)
        if not os.path.exists(full_path):
            print(f"ERROR: File not found: {full_path}")
            continue

        with open(full_path, "r", encoding="utf-8") as f:
            md_text = f.read()

        # Preprocess custom alert tags
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

        # Restore math blocks with HTML escaping for brackets
        for idx, block in enumerate(math_blocks):
            escaped_block = block.replace('<', '&lt;').replace('>', '&gt;')
            html_body = html_body.replace(f"MATHPLACEHOLDER{idx}ENDMATH", escaped_block)

        # Wrap in module container
        module_html = f"""
    <div class="module-container">
        <div style="background-color: #f8fafc; border-bottom: 2px solid #cbd5e1; padding: 12px 0; margin-bottom: 24px;">
            <span style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.1em; color: #64748b; font-weight: 600;">NLP Fundamentals Master Curriculum</span>
        </div>
        {html_body}
    </div>
"""
        modules_html.append(module_html)

    pygments_css = HtmlFormatter(style='monokai').get_style_defs('.codehilite')

    # Concatenate HTML
    body_content = "\n".join(modules_html)

    # Build full HTML template
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>NLP Fundamentals Master Study Guide</title>
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
        }}
        h2 {{
            font-size: 18px;
            color: #2563eb; /* Premium Royal Blue for H2 */
            margin-top: 24px;
            margin-bottom: 12px;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 4px;
            page-break-after: avoid;
        }}
        h3 {{
            font-size: 15px;
            color: #0284c7; /* Modern bright blue-teal for H3 */
            margin-top: 20px;
            margin-bottom: 10px;
            page-break-after: avoid;
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
            color: #3b82f6 !important; /* Royal blue bullet dot */
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
            color: #8b5cf6 !important; /* Purple circle for second level */
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
        
        code {{
            font-family: 'Consolas', 'Cascadia Code', 'Fira Code', 'Courier New', monospace;
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
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08);
            page-break-inside: avoid;
        }}
        
        /* Table Styling with Strict Cell Boundaries & wrapping */
        table {{
            width: 100% !important;
            border-collapse: collapse !important;
            margin: 20px 0 !important;
            font-size: 13px !important;
            border: 1.5px solid #64748b !important;
            page-break-inside: auto;
            table-layout: auto !important;
            max-width: 100% !important;
        }}
        tr {{
            page-break-inside: avoid;
            page-break-after: auto;
        }}
        th {{
            background-color: #0f172a !important;
            color: #f8fafc !important;
            font-weight: 700 !important;
            border: 1.5px solid #475569 !important;
            padding: 10px 12px !important;
            text-align: left !important;
        }}
        td {{
            border: 1px solid #cbd5e1 !important;
            padding: 9px 12px !important;
            text-align: left !important;
            vertical-align: top !important;
            word-wrap: break-word !important;
        }}
        tr:nth-child(even) td {{
            background-color: #f8fafc !important;
        }}
        
        /* Code Syntax Highlighting Styling & Transparency */
        {pygments_css}
        
        /* Prevent white background artifacts and dark text inside code blocks */
        .codehilite, 
        .codehilite pre, 
        .codehilite code, 
        .codehilite span,
        .codehilite div,
        .codehilite pre span,
        .codehilite pre code {{
            background-color: transparent !important;
            color: #f8fafc !important;
        }}
        
        .codehilite {{
            background-color: #0f172a !important;
            border-radius: 6px !important;
            padding: 14px 16px !important;
            margin: 16px 0 !important;
            border: 1px solid #334155 !important;
            max-width: 100% !important;
            overflow-x: hidden !important; /* Avoid horizontal scrollbar breaking PDF page scale */
        }}
        .codehilite pre {{
            background-color: transparent !important;
            color: #f8fafc !important;
            padding: 0 !important;
            margin: 0 !important;
            font-family: 'Consolas', 'Cascadia Code', 'Fira Code', 'Courier New', monospace !important;
            font-size: 12px !important;
            line-height: 1.5 !important;
            white-space: pre-wrap !important; /* Force line wrap in PDF */
            word-wrap: break-word !important;
            word-break: break-word !important;
            max-width: 100% !important;
        }}
        
        blockquote {{
            border-left: 4px solid #3b82f6;
            background-color: #eff6ff;
            margin: 16px 0;
            padding: 10px 16px;
            color: #1e40af;
            border-radius: 0 4px 4px 0;
        }}
        .katex-display {{
            margin: 16px 0 !important;
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
        MACHINE LEARNING & GENAI INTERVIEW PREPARATION
    </div>
    <h1 style="font-size: 30px; color: #0f172a; margin-bottom: 12px; line-height: 1.25; border: none; padding: 0;">
        Part 0: NLP Fundamentals: Vector Semantics, Sequence Modeling, and Classic Retrieval
    </h1>
    <h2 style="font-size: 18px; color: #3b82f6; font-weight: 500; margin-bottom: 40px; border: none;">
        Comprehensive Classical Foundations & Mathematical Study Guide
    </h2>
    <hr style="border: 0; height: 3px; background: linear-gradient(90deg, #3b82f6, #8b5cf6); width: 35%; margin: 30px auto; border-radius: 2px;" />
    
    <div style="margin-top: 60px; font-size: 13px; color: #475569; line-height: 1.8;">
        <p><b>Target Persona:</b> Senior AI Engineer / Applied AI Engineer (~3 YOE)</p>
        <p><b>Core Modules:</b> Preprocessing pipeline transitions, Subword Tokenizers (BPE/WordPiece/Unigram), TF-IDF matrix derivations, Feature Hashing, Classic Retrieval (Boolean/BM25), Statistical LMs, Word2Vec/GloVe/FastText embedding geometry, RNN vanishing gradient proofs, LSTM constant error carousel, Self-Attention scaling derivation ($1/\sqrt{{d_k}}$), evaluation loops, and 40 Q&As</p>
        <p><b>Includes:</b> BPE merge simulator, TF-IDF hand-calculations, Katz backoff probability derivations, scaled softmax dot-product variance proofs, and production error-correction loops.</p>
    </div>
</div>

<div class="toc-page" style="page-break-after: always; padding-top: 20px;">
    <h1 style="color: #0f172a; border-bottom: 2px solid #3b82f6; padding-bottom: 8px;">Table of Contents</h1>
    <ul style="list-style-type: none; padding-left: 0; margin-top: 20px;">
        <li style="margin-bottom: 12px; font-size: 14px; border-bottom: 1px dashed #e2e8f0; padding-bottom: 6px;"><a href="#module-01-nlp-introduction" style="text-decoration: none; color: #2563eb; font-weight: 600;">Module 01: NLP Introduction</a></li>
        <li style="margin-bottom: 12px; font-size: 14px; border-bottom: 1px dashed #e2e8f0; padding-bottom: 6px;"><a href="#module-02-text-preprocessing-subword-tokenization" style="text-decoration: none; color: #2563eb; font-weight: 600;">Module 02: Text Preprocessing & Subword Tokenization</a></li>
        <li style="margin-bottom: 12px; font-size: 14px; border-bottom: 1px dashed #e2e8f0; padding-bottom: 6px;"><a href="#module-03-text-representation-classical-retrieval-models" style="text-decoration: none; color: #2563eb; font-weight: 600;">Module 03: Text Representation & Classical Retrieval Models</a></li>
        <li style="margin-bottom: 12px; font-size: 14px; border-bottom: 1px dashed #e2e8f0; padding-bottom: 6px;"><a href="#module-04-statistical-language-models-smoothing" style="text-decoration: none; color: #2563eb; font-weight: 600;">Module 04: Statistical Language Models & Smoothing</a></li>
        <li style="margin-bottom: 12px; font-size: 14px; border-bottom: 1px dashed #e2e8f0; padding-bottom: 6px;"><a href="#module-05-word-embeddings-semantic-spaces" style="text-decoration: none; color: #2563eb; font-weight: 600;">Module 05: Word Embeddings & Semantic Spaces</a></li>
        <li style="margin-bottom: 12px; font-size: 14px; border-bottom: 1px dashed #e2e8f0; padding-bottom: 6px;"><a href="#module-06-sequence-models-recurrent-architectures" style="text-decoration: none; color: #2563eb; font-weight: 600;">Module 06: Sequence Models & Recurrent Architectures</a></li>
        <li style="margin-bottom: 12px; font-size: 14px; border-bottom: 1px dashed #e2e8f0; padding-bottom: 6px;"><a href="#module-07-attention-transformer-prerequisites" style="text-decoration: none; color: #2563eb; font-weight: 600;">Module 07: Attention & Transformer Prerequisites</a></li>
        <li style="margin-bottom: 12px; font-size: 14px; border-bottom: 1px dashed #e2e8f0; padding-bottom: 6px;"><a href="#module-08-nlp-evaluation-metrics-semantic-validation" style="text-decoration: none; color: #2563eb; font-weight: 600;">Module 08: NLP Evaluation Metrics & Semantic Validation</a></li>
        <li style="margin-bottom: 12px; font-size: 14px; border-bottom: 1px dashed #e2e8f0; padding-bottom: 6px;"><a href="#module-09-production-nlp-model-maintenance" style="text-decoration: none; color: #2563eb; font-weight: 600;">Module 09: Production NLP & Model Maintenance</a></li>
        <li style="margin-bottom: 12px; font-size: 14px; border-bottom: 1px dashed #e2e8f0; padding-bottom: 6px;"><a href="#module-10-nlp-fundamentals-high-frequency-interview-question-bank" style="text-decoration: none; color: #2563eb; font-weight: 600;">Module 10: NLP Fundamentals High-Frequency Interview Question Bank</a></li>
    </ul>
</div>

{body_content}

</body>
</html>
"""

    with open(html_out_path, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"Created master HTML file at: {html_out_path}")

    # Generate PDF using Microsoft Edge Headless CLI
    edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    temp_user_data = os.path.join(os.environ.get("TEMP", r"C:\Windows\Temp"), "edge_pdf_dir_tmp_nlp_master")
    
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
    
    print("Running Edge PDF compilation...")
    subprocess.run(cmd, capture_output=True, text=True)
    print(f"SUCCESS: Master PDF generated at: {pdf_out_path}")

def compile_cheatsheet():
    base_dir = r"d:\Study\Prep\machine-learning-prep\generative-ai-and-agentic-ai\00_nlp_fundamentals"
    md_path = os.path.join(base_dir, "modules", "nlp_interview_cheatsheet.md")
    html_out_path = os.path.join(base_dir, "nlp_interview_cheatsheet.html")
    pdf_out_path = os.path.join(base_dir, "nlp_interview_cheatsheet.pdf")
    
    if not os.path.exists(md_path):
        print(f"ERROR: Cheatsheet markdown not found: {md_path}")
        return
        
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()
        
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
        extensions=['fenced_code', 'tables', 'nl2br', 'sane_lists', 'codehilite']
    )

    # Restore math blocks with HTML escaping for brackets
    for idx, block in enumerate(math_blocks):
        escaped_block = block.replace('<', '&lt;').replace('>', '&gt;')
        html_body = html_body.replace(f"MATHPLACEHOLDER{idx}ENDMATH", escaped_block)
        
    # Styles for cheatsheet (clean, high density, fit-to-page margins)
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>NLP Fundamentals Interview Revision Cheatsheet</title>
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
            margin: 12mm 12mm 12mm 12mm;
        }}
        body {{
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
            font-size: 13px;
            line-height: 1.45;
            color: #334155;
            background-color: #ffffff;
            margin: 0;
            padding: 0;
        }}
        h1 {{
            font-size: 18px;
            color: #0f172a;
            border-bottom: 2px solid #3b82f6;
            padding-bottom: 4px;
            margin-top: 10px;
            margin-bottom: 12px;
            page-break-after: avoid;
        }}
        h2 {{
            font-size: 14px;
            color: #2563eb;
            margin-top: 14px;
            margin-bottom: 8px;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 2px;
            page-break-after: avoid;
        }}
        h3 {{
            font-size: 12px;
            color: #0284c7;
            margin-top: 10px;
            margin-bottom: 6px;
            page-break-after: avoid;
        }}
        p {{
            color: #334155;
            margin-bottom: 6px;
            font-size: 13px !important;
        }}
        
        /* Custom Bullet List and Ordered List for Cheatsheet */
        ul {{
            list-style: none !important;
            padding-left: 16px !important;
            margin-top: 4px !important;
            margin-bottom: 8px !important;
        }}
        ul li {{
            position: relative !important;
            margin-bottom: 5px !important;
            padding-left: 8px !important;
            font-size: 13px !important;
            line-height: 1.45 !important;
            color: #334155 !important;
        }}
        ul li::before {{
            content: "•" !important;
            color: #3b82f6 !important;
            font-weight: bold !important;
            font-size: 16px !important;
            display: inline-block !important;
            width: 1em !important;
            margin-left: -1em !important;
            position: absolute !important;
            top: -2px !important;
            left: 4px !important;
        }}
        ol {{
            padding-left: 16px !important;
            margin-top: 4px !important;
            margin-bottom: 8px !important;
        }}
        ol li {{
            margin-bottom: 5px !important;
            font-size: 13px !important;
            line-height: 1.45 !important;
            color: #334155 !important;
        }}
        
        code {{
            font-family: 'Consolas', 'Cascadia Code', 'Fira Code', 'Courier New', monospace;
            background-color: #f1f5f9;
            color: #0f172a;
            padding: 1px 4px;
            border-radius: 3px;
            font-size: 12px !important;
        }}
        table {{
            width: 100% !important;
            border-collapse: collapse !important;
            margin: 10px 0 !important;
            font-size: 11px !important;
            border: 1.5px solid #64748b !important;
            table-layout: auto !important;
        }}
        th {{
            background-color: #0f172a !important;
            color: #f8fafc !important;
            font-weight: 700 !important;
            border: 1.5px solid #475569 !important;
            padding: 6px 8px !important;
        }}
        td {{
            border: 1px solid #cbd5e1 !important;
            padding: 5px 8px !important;
            vertical-align: top !important;
        }}
        tr:nth-child(even) td {{
            background-color: #f8fafc !important;
        }}
        blockquote {{
            border-left: 3px solid #3b82f6;
            background-color: #eff6ff;
            margin: 10px 0;
            padding: 6px 12px;
            color: #1e40af;
            border-radius: 0 4px 4px 0;
        }}
        hr {{
            border: 0;
            height: 1px;
            background: #cbd5e1;
            margin: 15px 0;
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
    print(f"Created cheatsheet HTML file at: {html_out_path}")
    
    edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    temp_user_data = os.path.join(os.environ.get("TEMP", r"C:\Windows\Temp"), "edge_pdf_dir_tmp_nlp_cheatsheet")
    
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
    
    print("Running Edge PDF cheatsheet compilation...")
    subprocess.run(cmd, capture_output=True, text=True)
    print(f"SUCCESS: Cheatsheet PDF generated at: {pdf_out_path}")

if __name__ == "__main__":
    compile_master_guide()
    compile_cheatsheet()
