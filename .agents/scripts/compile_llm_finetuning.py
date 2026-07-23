import os
import re
import markdown
import subprocess
from pygments.formatters import HtmlFormatter

def compile_master_guide():
    base_dir = r"d:\Study\Prep\machine-learning-prep\generative-ai-and-agentic-ai\03_llm_finetuning_and_alignment"
    html_out_path = os.path.join(base_dir, "llm_finetuning_master_study_guide.html")
    pdf_out_path = os.path.join(base_dir, "llm_finetuning_master_study_guide.pdf")

    # Define the 9 modules in order
    module_files = [
        "01_finetuning_fundamentals.md",
        "02_parameter_efficient_finetuning.md",
        "03_instruction_tuning.md",
        "04_alignment.md",
        "05_preference_optimization.md",
        "06_model_merging_and_adapters.md",
        "07_production_considerations.md",
        "08_best_practices_and_common_failures.md",
        "09_interview_questions.md"
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
            <span style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.1em; color: #64748b; font-weight: 600;">LLM Fine-tuning & Alignment Master Curriculum</span>
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
<html>
<head>
    <meta charset="utf-8">
    <title>LLM Fine-tuning & Alignment Master Study Guide</title>
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
            margin: 18mm 15mm 18mm 15mm;
        }}
        body {{
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
            font-size: 13px;
            line-height: 1.6;
            color: #1e293b;
            background-color: #ffffff;
        }}
        
        /* Heading elements styling */
        h1 {{
            font-size: 20px;
            color: #0f172a;
            border-bottom: 1px solid #cbd5e1;
            padding-bottom: 6px;
            margin-top: 28px;
            margin-bottom: 12px;
            font-weight: 600;
        }}
        h2 {{
            font-size: 16px;
            color: #1e293b;
            margin-top: 22px;
            margin-bottom: 8px;
            font-weight: 600;
        }}
        h3 {{
            font-size: 14px;
            color: #334155;
            margin-top: 18px;
            margin-bottom: 6px;
            font-weight: 600;
        }}
        
        /* Table of Contents links */
        a {{
            color: #2563eb;
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}

        /* GFM standard native tables styling */
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 16px 0;
            font-size: 12px;
        }}
        th, td {{
            border: 1px solid #e2e8f0;
            padding: 8px 10px;
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

        /* Code Syntax Highlighting Styling & Transparency */
        {pygments_css}
        .codehilite {{
            background-color: #272822 !important;
            padding: 12px;
            border-radius: 6px;
            overflow-x: auto;
            margin: 14px 0;
        }}
        .codehilite pre, .codehilite code, .codehilite span {{
            background-color: transparent !important;
            color: #f8fafc !important;
            font-family: 'Consolas', 'Courier New', Courier, monospace !important;
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
    <h1 style="font-size: 30px; color: #0f172a; margin-bottom: 12px; line-height: 1.25; border: none; padding: 0;">
        Part 3: LLM Fine-tuning & Alignment: Adapting Pre-trained Models to Downstream Tasks
    </h1>
    <h2 style="font-size: 18px; color: #3b82f6; font-weight: 500; margin-bottom: 40px; border: none;">
        Comprehensive Production & Mathematical Study Guide
    </h2>
    <hr style="border: 0; height: 3px; background: linear-gradient(90deg, #3b82f6, #8b5cf6); width: 35%; margin: 30px auto; border-radius: 2px;" />
    
    <div style="margin-top: 60px; font-size: 13px; color: #475569; line-height: 1.8;">
        <p><b>Target Persona:</b> Senior AI Engineer / Applied AI Engineer (~3 YOE)</p>
        <p><b>Core Modules:</b> Fine-Tuning Fundamentals, PEFT (LoRA, DoRA, QLoRA), SFT Loss Masking, Preference Datasets, Preference Optimizations (DPO, ORPO, GRPO, KTO), Model Merging, Production serving, VRAM Sizing Calculations</p>
        <p><b>Includes:</b> VRAM Estimation Formulas, DPO/GRPO Loss Walkthroughs, custom HTML/CSS diagrams, PyTorch implementations, 25 Core Q&As</p>
    </div>
</div>

<div class="toc-page" style="page-break-after: always; padding-top: 20px;">
    <h1 style="color: #0f172a; border-bottom: 2px solid #3b82f6; padding-bottom: 8px;">Table of Contents</h1>
    <ul style="list-style-type: none; padding-left: 0; margin-top: 20px;">
        <li style="margin-bottom: 12px; font-size: 14px; border-bottom: 1px dashed #e2e8f0; padding-bottom: 6px;"><a href="#llm-fine-tuning-fundamentals" style="text-decoration: none; color: #2563eb; font-weight: 600;">Module 01: Fine-tuning Fundamentals</a></li>
        <li style="margin-bottom: 12px; font-size: 14px; border-bottom: 1px dashed #e2e8f0; padding-bottom: 6px;"><a href="#parameter-efficient-fine-tuning-peft" style="text-decoration: none; color: #2563eb; font-weight: 600;">Module 02: Parameter-Efficient Fine-Tuning (PEFT)</a></li>
        <li style="margin-bottom: 12px; font-size: 14px; border-bottom: 1px dashed #e2e8f0; padding-bottom: 6px;"><a href="#instruction-tuning-sft" style="text-decoration: none; color: #2563eb; font-weight: 600;">Module 03: Instruction Tuning & SFT</a></li>
        <li style="margin-bottom: 12px; font-size: 14px; border-bottom: 1px dashed #e2e8f0; padding-bottom: 6px;"><a href="#alignment-preference-datasets" style="text-decoration: none; color: #2563eb; font-weight: 600;">Module 04: Alignment & Preference Datasets</a></li>
        <li style="margin-bottom: 12px; font-size: 14px; border-bottom: 1px dashed #e2e8f0; padding-bottom: 6px;"><a href="#preference-optimization-direct-alignment" style="text-decoration: none; color: #2563eb; font-weight: 600;">Module 05: Preference Optimization (Direct Alignment)</a></li>
        <li style="margin-bottom: 12px; font-size: 14px; border-bottom: 1px dashed #e2e8f0; padding-bottom: 6px;"><a href="#model-merging-adapters" style="text-decoration: none; color: #2563eb; font-weight: 600;">Module 06: Model Merging & Adapters</a></li>
        <li style="margin-bottom: 12px; font-size: 14px; border-bottom: 1px dashed #e2e8f0; padding-bottom: 6px;"><a href="#production-considerations-serving" style="text-decoration: none; color: #2563eb; font-weight: 600;">Module 07: Production Considerations & Serving</a></li>
        <li style="margin-bottom: 12px; font-size: 14px; border-bottom: 1px dashed #e2e8f0; padding-bottom: 6px;"><a href="#best-practices-common-failures" style="text-decoration: none; color: #2563eb; font-weight: 600;">Module 08: Best Practices & Common Failures</a></li>
        <li style="margin-bottom: 12px; font-size: 14px; border-bottom: 1px dashed #e2e8f0; padding-bottom: 6px;"><a href="#llm-fine-tuning-alignment-high-frequency-question-bank" style="text-decoration: none; color: #2563eb; font-weight: 600;">Module 09: LLM Fine-Tuning & Alignment: High-Frequency Question Bank</a></li>
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
    temp_user_data = os.path.join(os.environ.get("TEMP", r"C:\Windows\Temp"), "edge_pdf_dir_tmp_ft_master")
    
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
