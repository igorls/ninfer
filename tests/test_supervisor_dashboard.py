"""Parse the JavaScript shipped in the Supervisor's executable HTML resource."""

import argparse
from html.parser import HTMLParser
from pathlib import Path
import subprocess


class Scripts(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.scripts = []
        self.current = None
        self.module = False

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            attrs = dict(attrs)
            self.current = [] if attrs.get("type", "") not in ("application/json", "importmap") else None
            self.module = attrs.get("type", "") == "module"

    def handle_data(self, data):
        if self.current is not None:
            self.current.append(data)

    def handle_endtag(self, tag):
        if tag == "script" and self.current is not None:
            self.scripts.append(("".join(self.current), self.module))
            self.current = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", required=True)
    parser.add_argument("--html", required=True, type=Path)
    args = parser.parse_args()
    html = Scripts()
    html.feed(args.html.read_text(encoding="utf-8"))
    if not html.scripts:
        raise AssertionError("Supervisor resource contains no JavaScript")
    for index, (script, module) in enumerate(html.scripts):
        command = [args.node, "--check", "-"]
        if module:
            command.insert(1, "--input-type=module")
        result = subprocess.run(command, input=script, text=True,
                                encoding="utf-8", capture_output=True, check=False)
        if result.returncode:
            raise AssertionError(f"Supervisor script {index} cannot parse:\n{result.stderr}")
    print(f"Parsed {len(html.scripts)} Supervisor scripts")


if __name__ == "__main__":
    main()
