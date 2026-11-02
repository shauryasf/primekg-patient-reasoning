#!/usr/bin/env python
"""Inline the model data and engine into demo/app.html.

Writes ``demo/index.html`` (a complete page that opens offline or on any static host) and
``demo/artifact.html`` (the same page body, for publishing where the host adds the document skeleton).
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    page = (HERE / "app.html").read_text(encoding="utf-8")
    data = (HERE / "model_data.json").read_text(encoding="utf-8").replace("</", "<\\/")
    engine = (HERE / "engine.js").read_text(encoding="utf-8")
    page = page.replace("/*__DATA__*/", data).replace("/*__ENGINE__*/", engine)
    (HERE / "artifact.html").write_text(page, encoding="utf-8")
    head, _, body = page.partition("</style>")
    full = f'<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n<style>body{{margin:0}}[hidden]{{display:none!important}}</style>\n{head}</style>\n</head>\n<body>\n{body}\n</body>\n</html>\n'
    (HERE / "index.html").write_text(full, encoding="utf-8")
    print(f"index.html {len(full) / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
