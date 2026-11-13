# Configuration file for the Sphinx documentation builder.
# Community documentation site (Khmer) for the upstream project
# rasbt/LLMs-from-scratch. Real information only; no credentials here.

project = "LLMs-from-scratch"
copyright = "2026 SeyhaLite (community translation); upstream code (c) Sebastian Raschka"
author = "SeyhaLite"

language = "km"

import os

# The sitemap extension is for HTML builds; it has no pages to process in linkcheck.
extensions = [] if os.environ.get("SPHINX_LINKCHECK") == "1" else ["sphinx_sitemap"]

# Primary host (GitHub Pages) — enables sitemap.xml and canonical URLs
html_baseurl = "https://seyhalite.github.io/LLMs-from-scratch/"
sitemap_url_scheme = "{link}"
sitemap_filename = "sitemap.xml"

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 2,
    "collapse_navigation": True,
    "logo_only": False,
    "prev_next_buttons_location": "bottom",
    "style_nav_header_background": "#1f2430",
}

html_show_sourcelink = False

html_title = "LLMs ពីមូលដ្ឋាន — ឯកសារសហគមន៍ភាសាខ្មែរ"
html_short_title = "LLMs ពីមូលដ្ឋាន"
html_logo = "_static/logo.svg"
html_favicon = "_static/logo.svg"

html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_extra_path = ["_extra"]

# Edit on GitHub (community repo)
html_context = {
    "km_baseurl": html_baseurl,
    "display_github": True,
    "github_user": "SeyhaLite",
    "github_repo": "LLMs-from-scratch",
    "github_version": "main",
    "conf_py_path": "/source/",
    "vcs_pageview_mode": "edit",
}
