"""MkDocs build hook: adds a visible 'Download notebook' link to every
rendered Jupyter notebook page, pointing at the raw .ipynb file mkdocs-jupyter
already copies alongside the rendered page (verified: same directory, same
base filename, .ipynb extension) -- so it works for every current and future
notebook with zero per-notebook manual list to maintain.
"""

import os


def on_page_content(html, page, config, files):
    src_path = getattr(page.file, "src_uri", "") or getattr(page.file, "src_path", "")
    if not src_path.lower().endswith(".ipynb"):
        return html

    notebook_filename = os.path.basename(src_path)
    banner = (
        '<div style="margin: 0 0 20px 0; padding: 10px 16px; '
        'background-color: var(--md-code-bg-color); border-left: 4px solid var(--md-primary-fg-color); '
        'border-radius: 0 4px 4px 0; font-size: 0.85rem;">'
        f'<a href="./{notebook_filename}" download '
        'style="text-decoration: none; font-weight: 600;">'
        "⬇ Download this notebook (.ipynb) to run locally</a>"
        "</div>"
    )
    return banner + html
