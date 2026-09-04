#!/usr/bin/env python3

import argparse
import json
import struct
import time
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


TRANSCRIBE_PROMPT = "Transcribe this audio. Return only the transcription."
TRANSLATE_PROMPT = (
    "Translate this audio into English. Return only the English translation."
)


class ContractServer(ThreadingHTTPServer):
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    def log_message(self, _format, *args):
        del args

    def send_json(self, status, payload):
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def reject(self, message):
        self.send_json(400, {"error": {"message": message}})

    def parse_multipart(self):
        content_type = self.headers.get("Content-Type", "")
        if not content_type.startswith("multipart/form-data;"):
            raise ValueError("request must use multipart/form-data")

        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        message = BytesParser(policy=policy.default).parsebytes(
            ("Content-Type: " + content_type + "\r\nMIME-Version: 1.0\r\n\r\n").encode()
            + body
        )
        if not message.is_multipart():
            raise ValueError("invalid multipart body")

        fields = {}
        files = {}
        for part in message.iter_parts():
            name = part.get_param("name", header="content-disposition")
            if not name:
                continue
            payload = part.get_payload(decode=True) or b""
            filename = part.get_filename()
            if filename:
                files[name] = (filename, part.get_content_type(), payload)
            else:
                fields[name] = payload.decode("utf-8")
        return fields, files

    def validate_contract(self, fields, files):
        expected_fields = {
            "response_format": "json",
            "temperature": "0",
            "max_tokens": "512",
        }
        for name, expected in expected_fields.items():
            if fields.get(name) != expected:
                raise ValueError(
                    f"field {name!r} must be {expected!r}, got {fields.get(name)!r}"
                )

        model = fields.get("model")
        if not model:
            raise ValueError("model field is missing")
        expected_prompt = (
            TRANSLATE_PROMPT if model == "translate-contract" else TRANSCRIBE_PROMPT
        )
        if fields.get("prompt") != expected_prompt:
            raise ValueError(
                f"prompt must be {expected_prompt!r}, got {fields.get('prompt')!r}"
            )
        if "language" in fields:
            raise ValueError("Gemma4 request must not force a source language")
        if "file" not in files:
            raise ValueError("file part is missing")

        filename, content_type, wav = files["file"]
        if filename != "audio.wav" or content_type != "audio/wav":
            raise ValueError("file metadata is invalid")
        if len(wav) < 44 or wav[:4] != b"RIFF" or wav[8:12] != b"WAVE":
            raise ValueError("file is not a WAV container")
        if struct.unpack_from("<H", wav, 22)[0] != 1:
            raise ValueError("WAV must be mono")
        if struct.unpack_from("<I", wav, 24)[0] != 16000:
            raise ValueError("WAV must be resampled to 16 kHz")
        if struct.unpack_from("<H", wav, 34)[0] != 16:
            raise ValueError("WAV must use 16-bit PCM")

    def do_POST(self):
        if self.path != "/v1/audio/transcriptions":
            self.reject("unexpected request path")
            return

        try:
            fields, files = self.parse_multipart()
            self.validate_contract(fields, files)
        except (UnicodeDecodeError, ValueError) as error:
            self.reject(str(error))
            return

        model = fields["model"]
        if model == "http-error":
            self.send_json(503, {"error": {"message": "contract failure"}})
        elif model == "malformed-response":
            self.send_json(200, {"result": "missing text"})
        elif model == "slow-response":
            time.sleep(2)
            self.send_json(200, {"text": "too late"})
        elif model == "translate-contract":
            self.send_json(200, {"text": "This is the English translation."})
        else:
            self.send_json(200, {"text": 'Hello "Gemma4"\n\u4e2d\u6587'})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port-file", required=True)
    args = parser.parse_args()

    server = ContractServer(("127.0.0.1", 0), Handler)
    Path(args.port_file).write_text(str(server.server_port), encoding="utf-8")
    server.serve_forever()


if __name__ == "__main__":
    main()
