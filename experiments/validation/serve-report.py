"""Researcher-only server for a built results report, on localhost (reach it through an SSH tunnel). Usage:
  python -I serve-report.py PORT REPORT_DIR EDITS_JSON
Serves REPORT_DIR with caching disabled. POST /edits (from the report's edit mode) replaces EDITS_JSON with the posted
{key: html} object and rewrites REPORT_DIR/edits.js; build-results-report.py writes the same edits.js from EDITS_JSON (spec field
EDITS), so saved text edits survive rebuilds. The report is private: never bind this to a public interface."""
import functools, http.server, json, sys
from pathlib import Path

MAX_BYTES = 8 << 20


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_POST(self):
        assert self.path == "/edits", self.path
        size = int(self.headers["Content-Length"])
        assert 0 < size <= MAX_BYTES, size
        edits = json.loads(self.rfile.read(size))
        assert isinstance(edits, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in edits.items())
        for path, text in ((EDITS, json.dumps(edits, indent=1, ensure_ascii=False)), (REPORT / "edits.js", "window.EDITS=" + json.dumps(edits, ensure_ascii=False) + ";")):
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(text)
            tmp.replace(path)
        self.send_response(204)
        self.end_headers()
        print(f"saved {len(edits)} edits to {EDITS}", flush=True)


PORT, REPORT, EDITS = int(sys.argv[1]), Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve()
assert (REPORT / "index.html").is_file() and EDITS.is_file(), (REPORT, EDITS)
http.server.ThreadingHTTPServer(("127.0.0.1", PORT), functools.partial(Handler, directory=str(REPORT))).serve_forever()
