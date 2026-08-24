import base64
import mimetypes
import os
import re
import shutil
import markdown
import subprocess
import tempfile
from pygments.formatters import HtmlFormatter

# Portable: derived from this script's own location (helpers/ -> topic root),
# never a hardcoded absolute path. See pdf_compiler/SKILL.md Section 4.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_browser_path():
    """Dynamically locate Edge or Chrome browser binary across standard paths."""
    candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
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

def fix_math_quotes(text):
    """Replace straight quotes with correct curly opening and closing quotes inside LaTeX text blocks."""
    def replace_in_math(math_match):
        math_content = math_match.group(0)
        def replace_in_text_cmd(text_match):
            text_content = text_match.group(0)
            text_content = text_content.replace('\\"', '"')
            parts = text_content.split('"')
            new_parts = []
            for i, part in enumerate(parts):
                if i == 0:
                    new_parts.append(part)
                elif i % 2 == 1:
                    new_parts.append('\u201c' + part)
                else:
                    new_parts.append('\u201d' + part)
            return "".join(new_parts)

        math_content = re.sub(r'\\text\{([^{}]+)\}', replace_in_text_cmd, math_content)
        return math_content

    text = re.sub(r'\$\$[\s\S]*?\$\$', replace_in_math, text)
    text = re.sub(r'(?<!\$)\$[^$\n]+\$(?!\$)', replace_in_math, text)
    return text

def transform_follow_up_questions(text):
    """Transform nested Common Interviewer Follow-Up Questions into flat, left-accented visual cards."""
    pattern = re.compile(
        r'^\s*[-*]\s+\*\*Common Interviewer Follow-Up Questions:\*\*\s*\n((?:\s{4,}.*\n?)*)',
        re.MULTILINE
    )

    def md_to_html_inline(val):
        val = re.sub(r'\*\*(.*?)\*\*|__(.*?)__', r'<strong>\1\2</strong>', val)
        val = re.sub(r'\*(.*?)\*|_(.*?)_', r'<em>\1\2</em>', val)
        return val

    def replacer(match):
        lines_block = match.group(1)
        q_a_pairs = []
        current_q = None
        current_a = []

        lines = lines_block.split('\n')
        for line in lines:
            stripped = line.strip()
            q_match = re.match(r'^\d+\.\s+\*Q:\*(.*)$', stripped)
            if not q_match:
                q_match = re.match(r'^\d+\.\s+Q:(.*)$', stripped)

            if q_match:
                if current_q:
                    q_a_pairs.append((current_q, " ".join(current_a)))
                current_q = q_match.group(1).strip()
                current_a = []
            elif stripped.startswith('*   *A:*') or stripped.startswith('* *A:*') or stripped.startswith('- *A:*') or re.match(r'^[-*]\s+\*A:\*', stripped):
                a_content = re.sub(r'^[-*]\s+\*A:\*', '', stripped).strip()
                current_a.append(a_content)
            elif stripped.startswith('*   A:') or stripped.startswith('* A:') or re.match(r'^[-*]\s+A:', stripped):
                a_content = re.sub(r'^[-*]\s+A:', '', stripped).strip()
                current_a.append(a_content)
            elif stripped:
                if current_q:
                    if line.startswith('        ') or line.startswith('\t\t'):
                        current_a.append(stripped.replace('*', '').strip())
                    else:
                        current_q += " " + stripped

        if current_q:
            q_a_pairs.append((current_q, " ".join(current_a)))

        html_out = [
            '<div class="follow-up-section" style="margin-top: 24px; margin-bottom: 24px; page-break-inside: avoid;">',
            '    <div style="font-weight: 700; color: #0f172a; border-bottom: 1px solid #cbd5e1; padding-bottom: 4px; margin-bottom: 12px; font-size: 15px; text-transform: uppercase; letter-spacing: 0.05em;">Common Interviewer Follow-Up Questions</div>'
        ]

        for q, a in q_a_pairs:
            q_html = md_to_html_inline(q)
            a_html = md_to_html_inline(a)
            html_out.append(f"""
    <div class="q-card" style="margin-bottom: 14px; border-left: 3px solid #3b82f6; padding-left: 12px; margin-top: 10px; page-break-inside: avoid;">
        <div style="font-weight: 600; color: #1e3a8a; font-size: 14.5px; margin-bottom: 4px;">Question: {q_html}</div>
        <div style="color: #334155; line-height: 1.5; font-size: 14px; text-align: justify !important;">{a_html}</div>
    </div>
""")
        html_out.append('</div>')
        return "\n".join(html_out) + "\n"

    return pattern.sub(replacer, text)

def transform_qa_module_followups(text):
    """Flatten Interview Q&A '#### Common Follow-up Questions' Q/A lists into left-accented cards.

    Distinct from transform_follow_up_questions() above, which handles the older
    '**Common Interviewer Follow-Up Questions:**' bullet style used in the theory modules.
    """
    pattern = re.compile(
        r'^#### Common Follow-up Questions\s*\n((?:.*\n?)*?)(?=\n---|\n#### |\Z)',
        re.MULTILINE
    )

    def md_to_html_inline(val):
        return re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', val)

    def replacer(match):
        block = match.group(1)
        q_a_pairs = []
        current_q, current_a = None, []
        for line in block.split('\n'):
            stripped = line.strip()
            q_match = re.match(r'^\d+\.\s+\*\*Q:\s*(.*?)\*\*\s*$', stripped)
            if q_match:
                if current_q:
                    q_a_pairs.append((current_q, " ".join(current_a)))
                current_q = q_match.group(1).strip()
                current_a = []
            elif re.match(r'^[-*]\s+\*\*A\*\*:', stripped):
                current_a.append(re.sub(r'^[-*]\s+\*\*A\*\*:\s*', '', stripped))
            elif stripped and current_q:
                current_a.append(stripped)
        if current_q:
            q_a_pairs.append((current_q, " ".join(current_a)))

        html_out = [
            '<div class="follow-up-section" style="margin-top: 20px; margin-bottom: 20px; page-break-inside: avoid;">',
            '    <div style="font-weight: 700; color: #0f172a; border-bottom: 1px solid #cbd5e1; '
            'padding-bottom: 4px; margin-bottom: 12px; font-size: 14.5px; text-transform: uppercase; '
            'letter-spacing: 0.05em;">Common Follow-up Questions</div>'
        ]
        for q, a in q_a_pairs:
            html_out.append(f"""
    <div class="q-card" style="margin-bottom: 12px; border-left: 3px solid #3b82f6; padding-left: 12px; page-break-inside: avoid;">
        <div style="font-weight: 600; color: #1e3a8a; font-size: 14px; margin-bottom: 4px;">Q: {md_to_html_inline(q)}</div>
        <div style="color: #334155; line-height: 1.5; font-size: 13.5px;">{md_to_html_inline(a)}</div>
    </div>
""")
        html_out.append('</div>\n')
        return "\n".join(html_out)

    return pattern.sub(replacer, text)

def embed_images_as_base64(text, plots_dir):
    """Rewrite '../plots/x.png' references to base64 data: URIs so the compiled HTML
    is fully portable (no machine-local file:// paths). See pdf_compiler/SKILL.md Section 4.
    """
    def replacer(match):
        rel_path = match.group(1)
        abs_path = os.path.normpath(os.path.join(plots_dir, os.path.basename(rel_path)))
        if not os.path.exists(abs_path):
            print(f"WARNING: image not found for embedding: {abs_path}")
            return match.group(0)
        mime, _ = mimetypes.guess_type(abs_path)
        mime = mime or "image/png"
        with open(abs_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")
        return f'(data:{mime};base64,{encoded})'

    return re.sub(r'\(\.\./plots/([^)]+)\)', replacer, text)

def preprocess_markdown(text, file_name):
    """Preprocess markdown list structures, frontmatter, relative paths, quotes, and Q&A formatting."""
    plots_dir = os.path.join(BASE_DIR, "plots")

    # 1. Transform follow-up questions to non-nested cards (both known formats)
    text = transform_follow_up_questions(text)
    text = transform_qa_module_followups(text)

    # 2. Embed plot images as base64 data URIs (portable HTML, no file:// paths)
    text = embed_images_as_base64(text, plots_dir)

    # 3. Strip frontmatter if present and replace it with Module number and title heading
    frontmatter_match = re.match(r'^---\n(.*?)\n---\n', text, re.DOTALL)
    if frontmatter_match:
        frontmatter_text = frontmatter_match.group(1)
        title_match = re.search(r'^title:\s*(.*)$', frontmatter_text, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else ""

        module_num_match = re.match(r'^(\d+)_', file_name)
        module_num = module_num_match.group(1) if module_num_match else ""

        text = text[frontmatter_match.end():]
        if module_num:
            text = f"# Module {module_num}: {title}\n\n" + text
        else:
            text = f"# {title}\n\n" + text

    # 4. Double the indentation of 2-space nested lists to 4-space
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

    # 5. Insert blank lines before list blocks if they are missing
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

def compile_document(md_files, html_out_path, pdf_out_path, page_title, header_label, subtitle, focus_areas, includes):
    base_dir = BASE_DIR
    alert_types = {
        'NOTE': ('#2563eb', '#eff6ff', '#1e40af', '\U0001F4CC NOTE'),
        'TIP': ('#059669', '#ecfdf5', '#065f46', '\U0001F4A1 TIP'),
        'IMPORTANT': ('#7c3aed', '#f5f3ff', '#5b21b6', '\u26A1 IMPORTANT'),
        'WARNING': ('#d97706', '#fffbeb', '#92400e', '\u26A0\uFE0F WARNING'),
        'CAUTION': ('#dc2626', '#fef2f2', '#991b1b', '\U0001F6A8 CAUTION')
    }

    modules_html = []

    for file_name in md_files:
        full_path = os.path.join(base_dir, "modules", file_name)
        if not os.path.exists(full_path):
            print(f"ERROR: File not found: {full_path}")
            continue

        with open(full_path, "r", encoding="utf-8") as f:
            md_text = f.read()

        md_text = fix_math_quotes(md_text)

        math_blocks = []
        def store_math(m):
            math_blocks.append(m.group(0))
            return f"MATHPLACEHOLDER{len(math_blocks)-1}ENDMATH"

        md_text = re.sub(r'\$\$[\s\S]*?\$\$', store_math, md_text)
        md_text = re.sub(r'(?<!\$)\$[^$\n]+\$(?!\$)', store_math, md_text)

        md_text = preprocess_markdown(md_text, file_name)

        for alert, (border, bg, color, label) in alert_types.items():
            pattern = re.compile(rf'>\s*\[!{alert}\]\s*\n((?:>[^\n]*\n?)*)', re.IGNORECASE)
            def alert_replacer(match, border=border, bg=bg, color=color, label=label):
                body_lines = match.group(1).replace('>', '').strip()
                return f'<div style="border-left: 4px solid {border}; background-color: {bg}; color: {color}; padding: 12px 16px; margin: 16px 0; border-radius: 0 6px 6px 0;"><strong>{label}:</strong><div style="margin-top: 4px;">{body_lines}</div></div>\n'
            md_text = pattern.sub(alert_replacer, md_text)

        html_body = markdown.markdown(
            md_text,
            extensions=['fenced_code', 'tables', 'toc', 'nl2br', 'sane_lists', 'codehilite']
        )

        for idx, block in enumerate(math_blocks):
            escaped_block = block.replace('<', '&lt;').replace('>', '&gt;')
            html_body = html_body.replace(f"MATHPLACEHOLDER{idx}ENDMATH", escaped_block)

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
        * {{
            scrollbar-width: none !important;
            -ms-overflow-style: none !important;
        }}
        ::-webkit-scrollbar {{
            display: none !important;
            width: 0 !important;
            height: 0 !important;
            background: transparent !important;
        }}
        body {{
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
            font-size: 14.5px;
            line-height: 1.6;
            color: #334155;
            background-color: #ffffff;
            margin: 0;
            padding: 0;
            text-align: justify !important;
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
            margin-bottom: 12px;
            text-align: justify !important;
        }}
        img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 24px auto;
            border-radius: 6px;
            border: 1px solid #cbd5e1;
        }}
        .module-container ul {{
            list-style: none !important;
            padding-left: 20px !important;
            margin-top: 6px !important;
            margin-bottom: 12px !important;
        }}
        .module-container ul > li {{
            position: relative !important;
            margin-bottom: 8px !important;
            padding-left: 10px !important;
            font-size: 14.5px !important;
            line-height: 1.6 !important;
            color: #334155 !important;
            text-align: justify !important;
        }}
        .module-container ul > li::before {{
            content: "\u2022" !important;
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
        .module-container ul ul > li::before {{
            content: "\u25E6" !important;
            color: #64748b !important;
            font-size: 16px !important;
            top: -1px !important;
        }}
        .module-container ol {{
            padding-left: 20px !important;
            margin-top: 6px !important;
            margin-bottom: 12px !important;
            list-style-type: decimal !important;
        }}
        .module-container ol > li {{
            margin-bottom: 8px !important;
            font-size: 14.5px !important;
            line-height: 1.6 !important;
            color: #334155 !important;
            list-style-type: decimal !important;
            text-align: justify !important;
        }}
        .module-container ol > li::before {{
            content: none !important;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 24px 0;
            font-size: 13.5px;
        }}
        th, td {{
            border: 1px solid #cbd5e1;
            padding: 10px 14px;
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
            white-space: pre-wrap !important;
            word-wrap: break-word !important;
            word-break: break-word !important;
        }}
        code {{
            padding: 2px 4px;
            font-size: 14.5px;
            color: #0f172a;
        }}
        pre code {{
            padding: 0;
            font-size: 13.5px;
            color: inherit;
            background-color: transparent;
            white-space: pre-wrap !important;
            word-wrap: break-word !important;
            word-break: break-word !important;
        }}
        {pygments_css}
        .codehilite {{
            background-color: #272822 !important;
            padding: 14px;
            border-radius: 6px;
            margin: 18px 0;
            overflow: visible !important;
        }}
        .codehilite pre {{
            margin: 0;
            background-color: transparent !important;
            white-space: pre-wrap !important;
            word-wrap: break-word !important;
            word-break: break-word !important;
        }}
        .codehilite code {{
            background-color: transparent !important;
            color: #f8f8f2 !important;
            white-space: pre-wrap !important;
            word-wrap: break-word !important;
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
            margin: 16px 0 !important;
            overflow-x: auto !important;
            overflow-y: hidden !important;
            scrollbar-width: none !important;
        }}
        .katex-display::-webkit-scrollbar {{
            display: none !important;
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
        {subtitle}
    </h2>
    <hr style="border: 0; height: 3px; background: linear-gradient(90deg, #3b82f6, #8b5cf6); width: 35%; margin: 30px auto; border-radius: 2px;" />

    <div style="margin-top: 60px; font-size: 14px; color: #475569; line-height: 1.8;">
        <p><b>Target Persona:</b> Senior AI Engineer / Applied AI Engineer (~3 YOE)</p>
        <p><b>Focus Areas:</b> {focus_areas}</p>
        <p><b>Includes:</b> {includes}</p>
    </div>
</div>

{body_content}

</body>
</html>
"""

    with open(html_out_path, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"Created HTML file at: {html_out_path}")

    # Generate PDF using Microsoft Edge Headless CLI.
    # Retry loop: headless Edge printing has been observed to exit 0 with no output
    # file on a transient race, then succeed immediately on retry. See pdf_compiler/SKILL.md
    # Section 4 for the confirmed-in-practice note on this.
    #
    # Cleanup uses manual mkdtemp + best-effort rmtree rather than
    # tempfile.TemporaryDirectory()'s context manager: Edge's Crashpad handler can
    # still hold a file lock inside the profile dir for a moment after the main
    # process exits, and strict cleanup raises WinError 145 ("directory not empty")
    # on that race. ignore_errors=True treats leftover profile dirs as harmless
    # (OS temp cleanup reclaims them later) instead of failing the whole compile.
    #
    # virtual-time-budget set to 30000ms: this topic has formula-bearing modules in
    # nearly every one of its 9 modules plus 7 inline SVG diagrams and 4 plots, so
    # the higher budget is used from the start rather than discovered the hard way.
    edge_path = get_browser_path()
    print(f"Running browser PDF compilation for {os.path.basename(pdf_out_path)}...")
    # Remove any stale PDF from a previous run BEFORE attempting to print. Without this,
    # a silently-failed print (exit 0, no write) leaves the OLD file sitting at pdf_out_path,
    # and the "exists and size > 0" check below would falsely report success on a file that
    # was never actually regenerated.
    if os.path.exists(pdf_out_path):
        os.remove(pdf_out_path)
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        # Kill any running msedge.exe immediately before this attempt -- confirmed in
        # practice: Edge's own background-mode/startup-boost service respawns a shadow
        # process within seconds of being killed, and ANY existing msedge.exe process
        # makes a new headless launch silently forward to that existing session instead
        # of running independently -- print-to-pdf then exits 0 with no file written.
        subprocess.run(["taskkill", "/F", "/IM", "msedge.exe"], capture_output=True, text=True)
        temp_user_data = tempfile.mkdtemp(prefix="edge_pdf_")
        try:
            cmd = [
                edge_path,
                f"--user-data-dir={temp_user_data}",
                "--headless",
                "--disable-gpu",
                "--run-all-compositor-stages-before-draw",
                "--virtual-time-budget=30000",
                "--no-pdf-header-footer",
                f"--print-to-pdf={pdf_out_path}",
                html_out_path
            ]
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        finally:
            shutil.rmtree(temp_user_data, ignore_errors=True)
        if os.path.exists(pdf_out_path) and os.path.getsize(pdf_out_path) > 0:
            print(f"SUCCESS: PDF generated at: {pdf_out_path} ({os.path.getsize(pdf_out_path)} bytes, attempt {attempt})")
            break
        print(f"WARNING: attempt {attempt}/{max_attempts} produced no valid PDF, retrying...")
    else:
        raise RuntimeError(f"PDF compilation did not produce a valid file at {pdf_out_path} after {max_attempts} attempts")

def main():
    base_dir = BASE_DIR

    print("\n--- Compiling Master LLM Inference & Optimization Study Guide ---")
    master_html = os.path.join(base_dir, "llm_inference_optimization_master_study_guide.html")
    master_pdf = os.path.join(base_dir, "llm_inference_optimization_master_study_guide.pdf")

    master_modules = [
        "01_inference_fundamentals_and_decoding_loop.md",
        "02_kv_cache_mechanics_and_memory_management.md",
        "03_pagedattention_and_memory_efficient_serving.md",
        "04_flashattention_and_io_aware_computation.md",
        "05_quantization_for_inference.md",
        "06_batching_strategies.md",
        "07_speculative_decoding_and_decoding_time_optimizations.md",
        "08_inference_serving_engines_and_production_architecture.md",
        "09_production_monitoring_cost_and_latency.md",
    ]
    compile_document(
        master_modules,
        master_html,
        master_pdf,
        "LLM Inference & Optimization",
        "LLM Inference & Optimization Curriculum Guide",
        subtitle="Decoding Mechanics, KV Cache, PagedAttention, FlashAttention, Quantization, Batching, Speculative Decoding &amp; Production Serving",
        focus_areas="Prefill/decode fundamentals, roofline model, KV cache &amp; GQA/MQA, PagedAttention, FlashAttention, inference quantization, static/continuous batching, speculative decoding, serving engines &amp; replica architecture, production monitoring &amp; cost modeling",
        includes="Step-by-Step Numerical Hand Calculations, Python Implementations, SVG Diagrams, Computed Plots",
    )

if __name__ == "__main__":
    main()
