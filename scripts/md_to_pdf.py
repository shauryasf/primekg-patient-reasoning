"""Convert a Markdown file to PDF using markdown + xhtml2pdf."""
import sys
import markdown
from xhtml2pdf import pisa
from pathlib import Path

def convert(md_path, pdf_path):
    text = Path(md_path).read_text(encoding='utf-8')
    html_body = markdown.markdown(text, extensions=['tables', 'fenced_code'])
    
    full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{ font-family: "Helvetica", "Arial", sans-serif; font-size: 11px; line-height: 1.6; margin: 40px; }}
  h1 {{ font-size: 22px; color: #222; border-bottom: 2px solid #333; padding-bottom: 6px; }}
  h2 {{ font-size: 18px; color: #333; border-bottom: 1px solid #999; padding-bottom: 4px; margin-top: 20px; }}
  h3 {{ font-size: 14px; color: #444; margin-top: 16px; }}
  code {{ background: #f4f4f4; padding: 1px 4px; border-radius: 3px; font-family: "Courier New", monospace; font-size: 10px; }}
  pre {{ background: #f4f4f4; padding: 10px; border-radius: 4px; font-size: 10px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
  th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: left; font-size: 10px; }}
  th {{ background-color: #e8e8e8; font-weight: bold; }}
  ul, ol {{ padding-left: 20px; }}
  li {{ margin-bottom: 4px; }}
  p {{ margin: 6px 0; }}
</style>
</head>
<body>
{html_body}
</body>
</html>"""
    
    with open(pdf_path, 'wb') as out:
        status = pisa.CreatePDF(full_html, dest=out, encoding='utf-8')
    
    if status.err:
        print(f"Error creating PDF: {status.err}")
        sys.exit(1)
    else:
        print(f"PDF created: {pdf_path}")

if __name__ == '__main__':
    convert(sys.argv[1], sys.argv[2])
