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
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>LLM Foundations Master Study Guide</title>
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
    <!-- Load Mermaid -->
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <script>
        document.addEventListener("DOMContentLoaded", function() {{
            const mermaidCodes = document.querySelectorAll("pre code.language-mermaid");
            mermaidCodes.forEach(codeEl => {{
                const preEl = codeEl.parentElement;
                const divEl = document.createElement("div");
                divEl.className = "mermaid";
                divEl.textContent = codeEl.textContent;
                preEl.replaceWith(divEl);
            }});
            mermaid.initialize({{
                startOnLoad: true,
                securityLevel: 'loose',
                theme: 'neutral'
            }});
        }});
    </script>
    <style>
        @page {{
            size: A4;
            margin: 18mm 15mm 18mm 15mm;
        }}
        body {{
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
            font-size: 13px;
            line-height: 1.6;
            color: #1e293b;
            background-color: #ffffff;
            margin: 0;
            padding: 0;
        }}
        h1 {{
            font-size: 22px;
            color: #0f172a;
            border-bottom: 2px solid #3b82f6;
            padding-bottom: 6px;
            margin-top: 24px;
            margin-bottom: 14px;
        }}
        h2 {{
            font-size: 17px;
            color: #1e40af;
            margin-top: 20px;
            margin-bottom: 10px;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 4px;
        }}
        h3 {{
            font-size: 15px;
            color: #0369a1;
            margin-top: 16px;
            margin-bottom: 8px;
        }}
        
        p, li {{
            color: #334155;
            margin-bottom: 8px;
        }}
        
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
        .codehilite .hll {{ background-color: #49483e }}
        .codehilite {{ background: #272822; color: #f8f8f2 }}
        .codehilite .c {{ color: #75715e }} /* Comment */
        .codehilite .err {{ color: #960050; background-color: #1e0010 }} /* Error */
        .codehilite .esc {{ color: #f8f8f2 }} /* Escape */
        .codehilite .g {{ color: #f8f8f2 }} /* Generic */
        .codehilite .k {{ color: #66d9ef }} /* Keyword */
        .codehilite .l {{ color: #ae81ff }} /* Literal */
        .codehilite .n {{ color: #f8f8f2 }} /* Name */
        .codehilite .o {{ color: #f92672 }} /* Operator */
        .codehilite .x {{ color: #f8f8f2 }} /* Other */
        .codehilite .p {{ color: #f8f8f2 }} /* Punctuation */
        .codehilite .ch {{ color: #75715e }} /* Comment.Hashbang */
        .codehilite .cm {{ color: #75715e }} /* Comment.Multiline */
        .codehilite .cp {{ color: #75715e }} /* Comment.Preproc */
        .codehilite .cpf {{ color: #75715e }} /* Comment.PreprocFile */
        .codehilite .c1 {{ color: #75715e }} /* Comment.Single */
        .codehilite .cs {{ color: #75715e }} /* Comment.Special */
        .codehilite .gd {{ color: #f92672 }} /* Generic.Deleted */
        .codehilite .ge {{ color: #f8f8f2; font-style: italic }} /* Generic.Emph */
        .codehilite .gr {{ color: #f8f8f2 }} /* Generic.Error */
        .codehilite .gh {{ color: #f8f8f2 }} /* Generic.Heading */
        .codehilite .gi {{ color: #a6e22e }} /* Generic.Inserted */
        .codehilite .go {{ color: #66d9ef }} /* Generic.Output */
        .codehilite .gp {{ color: #f92672; font-weight: bold }} /* Generic.Prompt */
        .codehilite .gs {{ color: #f8f8f2; font-weight: bold }} /* Generic.Strong */
        .codehilite .gu {{ color: #75715e }} /* Generic.Subheading */
        .codehilite .gt {{ color: #f8f8f2 }} /* Generic.Traceback */
        .codehilite .kc {{ color: #66d9ef }} /* Keyword.Constant */
        .codehilite .kd {{ color: #66d9ef }} /* Keyword.Declaration */
        .codehilite .kn {{ color: #f92672 }} /* Keyword.Namespace */
        .codehilite .kp {{ color: #66d9ef }} /* Keyword.Pseudo */
        .codehilite .kr {{ color: #66d9ef }} /* Keyword.Reserved */
        .codehilite .kt {{ color: #66d9ef }} /* Keyword.Type */
        .codehilite .ld {{ color: #e6db74 }} /* Literal.Date */
        .codehilite .m {{ color: #ae81ff }} /* Literal.Number */
        .codehilite .s {{ color: #e6db74 }} /* Literal.String */
        .codehilite .na {{ color: #a6e22e }} /* Name.Attribute */
        .codehilite .nb {{ color: #f8f8f2 }} /* Name.Builtin */
        .codehilite .nc {{ color: #a6e22e }} /* Name.Class */
        .codehilite .no {{ color: #66d9ef }} /* Name.Constant */
        .codehilite .nd {{ color: #a6e22e }} /* Name.Decorator */
        .codehilite .ni {{ color: #f8f8f2 }} /* Name.Entity */
        .codehilite .ne {{ color: #a6e22e }} /* Name.Exception */
        .codehilite .nf {{ color: #a6e22e }} /* Name.Function */
        .codehilite .nl {{ color: #f8f8f2 }} /* Name.Label */
        .codehilite .nn {{ color: #f8f8f2 }} /* Name.Namespace */
        .codehilite .nx {{ color: #a6e22e }} /* Name.Other */
        .codehilite .py {{ color: #f8f8f2 }} /* Name.Property */
        .codehilite .nt {{ color: #f92672 }} /* Name.Tag */
        .codehilite .nv {{ color: #f8f8f2 }} /* Name.Variable */
        .codehilite .ow {{ color: #f92672 }} /* Operator.Word */
        .codehilite .pm {{ color: #f8f8f2 }} /* Punctuation.Marker */
        .codehilite .w {{ color: #f8f8f2 }} /* Text.Whitespace */
        .codehilite .mb {{ color: #ae81ff }} /* Literal.Number.Bin */
        .codehilite .mf {{ color: #ae81ff }} /* Literal.Number.Float */
        .codehilite .mh {{ color: #ae81ff }} /* Literal.Number.Hex */
        .codehilite .mi {{ color: #ae81ff }} /* Literal.Number.Integer */
        .codehilite .mo {{ color: #ae81ff }} /* Literal.Number.Oct */
        .codehilite .sa {{ color: #e6db74 }} /* Literal.String.Affix */
        .codehilite .sb {{ color: #e6db74 }} /* Literal.String.Backtick */
        .codehilite .sc {{ color: #e6db74 }} /* Literal.String.Char */
        .codehilite .dl {{ color: #e6db74 }} /* Literal.String.Delimiter */
        .codehilite .sd {{ color: #e6db74 }} /* Literal.String.Doc */
        .codehilite .s2 {{ color: #e6db74 }} /* Literal.String.Double */
        .codehilite .se {{ color: #ae81ff }} /* Literal.String.Escape */
        .codehilite .sh {{ color: #e6db74 }} /* Literal.String.Heredoc */
        .codehilite .si {{ color: #e6db74 }} /* Literal.String.Interpol */
        .codehilite .sx {{ color: #e6db74 }} /* Literal.String.Other */
        .codehilite .sr {{ color: #e6db74 }} /* Literal.String.Regex */
        .codehilite .s1 {{ color: #e6db74 }} /* Literal.String.Single */
        .codehilite .ss {{ color: #e6db74 }} /* Literal.String.Symbol */
        .codehilite .bp {{ color: #f8f8f2 }} /* Name.Builtin.Pseudo */
        .codehilite .fm {{ color: #a6e22e }} /* Name.Function.Magic */
        .codehilite .vc {{ color: #f8f8f2 }} /* Name.Variable.Class */
        .codehilite .vg {{ color: #f8f8f2 }} /* Name.Variable.Global */
        .codehilite .vi {{ color: #f8f8f2 }} /* Name.Variable.Instance */
        .codehilite .vm {{ color: #f8f8f2 }} /* Name.Variable.Magic */
        .codehilite .il {{ color: #ae81ff }} /* Literal.Number.Integer.Long */

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
        Comprehensive Production & Mathematical Study Guide
    </h2>
    <hr style="border: 0; height: 3px; background: linear-gradient(90deg, #3b82f6, #8b5cf6); width: 35%; margin: 30px auto; border-radius: 2px;" />
    
    <div style="margin-top: 60px; font-size: 14px; color: #475569; line-height: 1.8;">
        <p><b>Target Persona:</b> Senior AI Engineer / Applied AI Engineer (~3 YOE)</p>
        <p><b>Core Modules:</b> Evolution of NLP, Transformer Layers, Attention Math, Caching, Tokenization, LLM Architectures, Scaling & MoE, Limitations, Serving & Optimization</p>
        <p><b>Includes:</b> 11 Essential Formulas, Complexity Analyses, GFM Tables with Boundaries, Monokai Code Highlighting, PyTorch Code, Standardized Interview Q&A</p>
    </div>
</div>

<div class="toc-page" style="page-break-after: always; padding-top: 20px;">
    <h1 style="color: #0f172a; border-bottom: 2px solid #3b82f6; padding-bottom: 8px;">Table of Contents</h1>
    <ul style="list-style-type: none; padding-left: 0; margin-top: 20px;">
        <li style="margin-bottom: 12px; font-size: 14px; border-bottom: 1px dashed #e2e8f0; padding-bottom: 6px;"><a href="#module-01-evolution-of-nlp-from-bow-to-transformers" style="text-decoration: none; color: #2563eb; font-weight: 600;">Module 01: Evolution of NLP: From BoW to Transformers</a></li>
        <li style="margin-bottom: 12px; font-size: 14px; border-bottom: 1px dashed #e2e8f0; padding-bottom: 6px;"><a href="#module-02-transformer-architecture" style="text-decoration: none; color: #2563eb; font-weight: 600;">Module 02: Transformer Architecture</a></li>
        <li style="margin-bottom: 12px; font-size: 14px; border-bottom: 1px dashed #e2e8f0; padding-bottom: 6px;"><a href="#module-03-self-attention-deep-dive" style="text-decoration: none; color: #2563eb; font-weight: 600;">Module 03: Self-Attention Deep Dive</a></li>
        <li style="margin-bottom: 12px; font-size: 14px; border-bottom: 1px dashed #e2e8f0; padding-bottom: 6px;"><a href="#module-04-positional-encodings" style="text-decoration: none; color: #2563eb; font-weight: 600;">Module 04: Positional Encodings</a></li>
        <li style="margin-bottom: 12px; font-size: 14px; border-bottom: 1px dashed #e2e8f0; padding-bottom: 6px;"><a href="#module-05-tokenization-and-embeddings" style="text-decoration: none; color: #2563eb; font-weight: 600;">Module 05: Tokenization and Embeddings</a></li>
        <li style="margin-bottom: 12px; font-size: 14px; border-bottom: 1px dashed #e2e8f0; padding-bottom: 6px;"><a href="#module-06-context-window-and-kv-cache" style="text-decoration: none; color: #2563eb; font-weight: 600;">Module 06: Context Window and KV Cache</a></li>
        <li style="margin-bottom: 12px; font-size: 14px; border-bottom: 1px dashed #e2e8f0; padding-bottom: 6px;"><a href="#module-07-decoding-strategies" style="text-decoration: none; color: #2563eb; font-weight: 600;">Module 07: Decoding Strategies</a></li>
        <li style="margin-bottom: 12px; font-size: 14px; border-bottom: 1px dashed #e2e8f0; padding-bottom: 6px;"><a href="#module-08-modern-llm-architectures" style="text-decoration: none; color: #2563eb; font-weight: 600;">Module 08: Modern LLM Architectures</a></li>
        <li style="margin-bottom: 12px; font-size: 14px; border-bottom: 1px dashed #e2e8f0; padding-bottom: 6px;"><a href="#module-09-scaling-laws-and-moe" style="text-decoration: none; color: #2563eb; font-weight: 600;">Module 09: Scaling Laws and Mixture of Experts (MoE)</a></li>
        <li style="margin-bottom: 12px; font-size: 14px; border-bottom: 1px dashed #e2e8f0; padding-bottom: 6px;"><a href="#module-10-llm-limitations" style="text-decoration: none; color: #2563eb; font-weight: 600;">Module 10: LLM Limitations</a></li>
        <li style="margin-bottom: 12px; font-size: 14px; border-bottom: 1px dashed #e2e8f0; padding-bottom: 6px;"><a href="#module-11-llm-inference-pipeline" style="text-decoration: none; color: #2563eb; font-weight: 600;">Module 11: LLM Inference Pipeline</a></li>
        <li style="margin-bottom: 12px; font-size: 14px; border-bottom: 1px dashed #e2e8f0; padding-bottom: 6px;"><a href="#module-12-llm-foundations-master-interview-prep-cheatsheet" style="text-decoration: none; color: #2563eb; font-weight: 600;">Module 12: LLM Foundations Master Interview Prep & Cheatsheet</a></li>
        <li style="margin-bottom: 12px; font-size: 14px; border-bottom: 1px dashed #e2e8f0; padding-bottom: 6px;"><a href="#module-13-llm-foundations-interview-questions-answers" style="text-decoration: none; color: #2563eb; font-weight: 600;">Module 13: LLM Foundations Interview Questions & Answers</a></li>
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

