import os
import re
import markdown
import subprocess
from pygments.formatters import HtmlFormatter

def compile_master_guide():
    base_dir = r"d:\Study\Prep\machine-learning-prep\generative-ai-and-agentic-ai\00_llm_foundations"
    html_out_path = os.path.join(base_dir, "llm_foundations_master_study_guide.html")
    pdf_out_path = os.path.join(base_dir, "llm_foundations_master_study_guide.pdf")

    # Define the 13 modules in order
    module_files = [
        "01_evolution_of_nlp.md",
        "02_transformer_architecture.md",
        "03_self_attention_deep_dive.md",
        "04_positional_encodings.md",
        "05_tokenization_and_embeddings.md",
        "06_context_window_and_kv_cache.md",
        "07_decoding_strategies.md",
        "08_modern_llm_architectures.md",
        "09_scaling_laws_and_moe.md",
        "10_llm_limitations.md",
        "11_llm_inference_pipeline.md",
        "12_llm_foundations_interview_masterclass.md",
        "13_interview_questions.md"
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
        full_path = os.path.join(base_dir, file_name)
        if not os.path.exists(full_path):
            print(f"ERROR: File not found: {full_path}")
            continue

        with open(full_path, "r", encoding="utf-8") as f:
            md_text = f.read()

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
            <span style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.1em; color: #64748b; font-weight: 600;">LLM Foundations Master Curriculum</span>
        </div>
        {html_body}
    </div>
"""
        modules_html.append(module_html)

    pygments_css = HtmlFormatter(style='monokai').get_style_defs('.codehilite')

    # Concatenate HTML
    body_content = "\n".join(modules_html)

    full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>LLM Foundations Master Study Guide</title>
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
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            color: #0f172a;
            line-height: 1.6;
            font-size: 14px;
            padding: 40px;
            max-width: 900px;
            margin: 0 auto;
        }}
        h1, h2, h3, h4 {{
            color: #0f172a;
            font-weight: 700;
            margin-top: 24px;
            margin-bottom: 12px;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 6px;
        }}
        h1 {{ font-size: 24px; }}
        h2 {{ font-size: 20px; }}
        h3 {{ font-size: 16px; }}
        
        p {{ margin-bottom: 14px; }}
        
        code {{
            font-family: 'Consolas', 'Cascadia Code', 'Fira Code', 'Courier New', monospace;
            background-color: #f1f5f9;
            color: #0f172a;
            padding: 2px 5px;
            border-radius: 4px;
            font-size: 12px;
        }}
        
        /* Table Styling with Strict Cell Boundaries */
        table {{
            width: 100% !important;
            border-collapse: collapse !important;
            margin: 16px 0 !important;
            font-size: 11.5px !important;
            border: 1.5px solid #64748b !important;
            page-break-inside: auto;
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
            padding: 8px 10px !important;
            text-align: left !important;
        }}
        td {{
            border: 1px solid #cbd5e1 !important;
            padding: 7px 10px !important;
            text-align: left !important;
            vertical-align: top !important;
        }}
        tr:nth-child(even) td {{
            background-color: #f8fafc !important;
        }}

        /* Pygments Code Syntax Highlighting Theme (Monokai Dark) */
        pre {{ line-height: 125%; }}
        td.linenos .normal {{ color: inherit; background-color: transparent; padding-left: 5px; padding-right: 5px; }}
        span.linenos {{ color: inherit; background-color: transparent; padding-left: 5px; padding-right: 5px; }}
        td.linenos .special {{ color: #000000; background-color: #ffffc0; padding-left: 5px; padding-right: 5px; }}
        span.linenos.special {{ color: #000000; background-color: #ffffc0; padding-left: 5px; padding-right: 5px; }}
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
            padding: 12px 14px !important;
            margin: 14px 0 !important;
            overflow-x: auto !important;
            border: 1px solid #334155 !important;
        }}
        .codehilite pre {{
            background-color: transparent !important;
            color: #f8fafc !important;
            padding: 0 !important;
            margin: 0 !important;
            font-family: 'Consolas', 'Cascadia Code', 'Fira Code', 'Courier New', monospace !important;
            font-size: 11px !important;
            line-height: 1.45 !important;
            white-space: pre-wrap !important;
            word-break: break-word !important;
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
        MACHINE LEARNING & GENAI INTERVIEW PREPARATION
    </div>
    <h1 style="font-size: 34px; color: #0f172a; margin-bottom: 12px; line-height: 1.25; border: none; padding: 0;">
        Part 1: LLM Foundations: From NLP to Transformer Infrastructures
    </h1>
    <h2 style="font-size: 20px; color: #3b82f6; font-weight: 500; margin-bottom: 40px; border: none;">
        Comprehensive Technical & Mathematical Study Guide
    </h2>
    <hr style="border: 0; height: 3px; background: linear-gradient(90deg, #3b82f6, #8b5cf6); width: 35%; margin: 30px auto; border-radius: 2px;" />
    <p style="font-size: 14px; color: #64748b; margin-top: 40px;">
        Curated Study Guide Covering Evolution of NLP, Transformer Layers, Attention Math, Caching, Tokenization, and LLM Inference Pipelines.
    </p>
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
    temp_user_data = os.path.join(os.environ.get("TEMP", r"C:\Windows\Temp"), "edge_pdf_dir_tmp_master")
    
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

if __name__ == "__main__":
    compile_master_guide()
