import base64
import sys
import tempfile
import webbrowser
from pathlib import Path


def show_html_in_notebook(html: str, stem: str = "viz", height: int = 600) -> Path:
    """Render HTML inline in notebooks and save it to a temp file.

    Returns the saved HTML path so callers can reuse it if needed.
    """
    path = Path(tempfile.mkdtemp()) / f"{stem}.html"
    path.write_text(html, encoding="utf-8")

    if "ipykernel" in sys.modules:
        from IPython.display import HTML, display

        encoded = base64.b64encode(html.encode("utf-8")).decode("ascii")
        data_uri = f"data:text/html;base64,{encoded}"
        display(
            HTML(
                f'<iframe src="{data_uri}" width="100%" height="{height}" '
                f'style="border:1px solid #ccc; border-radius:4px;"></iframe>'
                f'<p style="margin-top:4px;font-size:12px;color:#666;">'
                f"Saved to <code>{path}</code></p>"
            )
        )
    else:
        print(f"Saved: {path}")
        webbrowser.open(path.as_uri())

    return path
