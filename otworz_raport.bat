@echo off
cd /d "%~dp0"
py -c "
import os, webbrowser, markdown

script_dir = r'%~dp0'
src = os.path.join(script_dir, 'raport.md')
dst = os.path.join(script_dir, 'raport.html')

with open(src, encoding='utf-8') as f:
    md_text = f.read()

html_body = markdown.markdown(md_text, extensions=['tables', 'fenced_code'])

html = '''<!DOCTYPE html>
<html lang=\"pl\">
<head>
<meta charset=\"UTF-8\">
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
<title>Raport – Prognoza Zuzycia Energii</title>
<style>
  body { font-family: Segoe UI, Arial, sans-serif; max-width: 960px; margin: 40px auto; padding: 0 24px; color: #222; line-height: 1.7; }
  h1 { color: #1a5c96; border-bottom: 3px solid #1a5c96; padding-bottom: 8px; }
  h2 { color: #1a5c96; border-bottom: 1px solid #cce; padding-bottom: 4px; margin-top: 2em; }
  h3 { color: #2c3e50; margin-top: 1.6em; }
  table { border-collapse: collapse; width: 100%%; margin: 1em 0; }
  th { background: #1a5c96; color: white; padding: 8px 12px; text-align: left; }
  td { padding: 7px 12px; border-bottom: 1px solid #dde; }
  tr:nth-child(even) td { background: #f4f8ff; }
  code { background: #f0f0f0; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }
  pre { background: #f0f0f0; padding: 16px; border-radius: 6px; overflow-x: auto; }
  pre code { background: none; padding: 0; }
  blockquote { border-left: 4px solid #1a5c96; margin: 0; padding: 8px 16px; background: #f0f5ff; color: #444; }
  hr { border: none; border-top: 2px solid #dde; margin: 2em 0; }
</style>
</head>
<body>
''' + html_body + '''
</body>
</html>'''

with open(dst, 'w', encoding='utf-8') as f:
    f.write(html)

webbrowser.open('file:///' + dst.replace('\\\\', '/'))
print('Otwarto raport.html w przegladarce.')
"
