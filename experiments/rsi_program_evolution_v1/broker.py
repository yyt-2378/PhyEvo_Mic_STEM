"""Copy one sealed request into a neutral transport packet; publish raw answers.

No API credentials, data arrays or historical conversations are read. A broker
does not alter the proposal or execute its source. All files are exclusive new
artifacts; packet paths reveal neither experimental arm nor source names.
"""
import argparse
import json
from pathlib import Path

from .contracts import file_hash, inside, write_new
from .provider import publish_response


ROOT = Path(__file__).resolve().parents[2]


def stage(source, packet):
    source, packet = inside(ROOT, source), inside(ROOT, packet)
    request = inside(ROOT, source / "request.json")
    packet.mkdir(parents=True, exist_ok=False)
    with (packet / "request.json").open("xb") as stream:
        stream.write(request.read_bytes())
    write_new(packet / "transport.json", {"source": str(source.relative_to(ROOT)),
              "request_sha256": file_hash(request), "packet_sha256": file_hash(packet / "request.json"),
              "transport_only": True})
    print(json.dumps({"request": str(packet / "request.json"), "sha256": file_hash(request)}))


def publish(packet, origin):
    packet = inside(ROOT, packet)
    info = json.loads((packet / "transport.json").read_text())
    source = inside(ROOT, info["source"])
    if any(file_hash(path) != info["request_sha256"]
           for path in (source / "request.json", packet / "request.json")):
        raise ValueError("Sealed transport input changed")
    request = json.loads((packet / "request.json").read_text())
    digest = publish_response(source, raw_text=(packet / "raw_response.json").read_text(),
                              model=request["model"], origin=origin)
    write_new(packet / "published.json", {"origin": origin, "response_sha256": digest,
              "raw_response_sha256": file_hash(packet / "raw_response.json"), "usage": None})
    print(json.dumps({"published": digest, "origin": origin}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("stage", "publish"))
    parser.add_argument("packet")
    parser.add_argument("--source")
    parser.add_argument("--origin")
    args = parser.parse_args()
    if args.command == "stage":
        if not args.source:
            parser.error("stage requires --source")
        stage(args.source, args.packet)
    else:
        if not args.origin:
            parser.error("publish requires --origin")
        publish(args.packet, args.origin)
