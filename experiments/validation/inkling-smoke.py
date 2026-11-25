"""Smoke test of a live local_openai service (scripts/inkling-service.slurm) through the real InferenceBroker.

Run on a CPU compute node (test/sapphire), as the strict controller would reach the GPU service:
  PYTHONPATH=src .venv/bin/python experiments/validation/inkling-smoke.py RECORD OUT.json
Checks a tool call, a tool-result round trip, an image question and every reasoning effort through the broker; then raw
requests measure decode throughput (1 and 5 streams), a long-context needle at ~120K prompt tokens and its prefix-cache reuse.
Prints and writes the measurements; any failed check raises."""
import base64
import json
import struct
import sys
import time
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor

from sim2sim.runtime import InferenceBroker, local_service

MODEL = "inkling-2-mini"
BASH = [{"name": "bash", "description": "Run a shell command and return its output.",
         "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}]


def png(width, height, pixel):
    rows = b"".join(b"\0" + b"".join(bytes(pixel(x, y)) for x in range(width)) for y in range(height))
    chunk = lambda kind, data: struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def main(record, out):
    endpoint, token = local_service(record, MODEL)
    results = {"service": {k: v for k, v in json.load(open(record)).items() if k != "token"}}

    def call(messages, tools=BASH, effort="medium", max_tokens=4096):
        broker = InferenceBroker({"api": "local_openai", "endpoint": record, "model": MODEL, "reasoning_effort": effort,
                                  "max_output_tokens": max_tokens, "timeout_seconds": 900, "max_calls": 100})
        started = time.monotonic()
        reply = broker.call({"messages": messages, "tools": tools}, 900)
        return reply, round(time.monotonic() - started, 2)

    def raw(body):
        request = urllib.request.Request(endpoint, data=json.dumps({"model": MODEL, "stream": False, **body}).encode(),
                                         headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})
        started = time.monotonic()
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=1800) as response:
            reply = json.load(response)
        return reply, time.monotonic() - started

    def save():
        with open(out, "w") as handle:
            json.dump(results, handle, indent=1)

    system = {"role": "system", "content": "You are a careful agent. Use the bash tool to act."}
    ask = {"role": "user", "content": "Read the file /data/run/answer.txt with the bash tool and tell me the number it contains."}
    reply, seconds = call([system, ask])
    results["tool_call"] = {"seconds": seconds, "content": reply["content"][:500], "tool_calls": reply["tool_calls"],
                            "finish_reason": reply["finish_reason"], "usage": reply["usage"]}
    save()
    assert reply["tool_calls"] and reply["tool_calls"][0]["name"] == "bash", reply
    json.loads(reply["tool_calls"][0]["arguments"])["command"]

    # Answer every bash call with the file's content until the model reports in text (it may re-check with another command).
    messages, results["round_trip"] = [system, ask], []
    while reply["tool_calls"]:
        messages += [{"role": "assistant", "content": reply["content"], "tool_calls": reply["tool_calls"]}]
        messages += [{"role": "tool", "tool_call_id": c["id"], "content": "4817\n"} for c in reply["tool_calls"]]
        assert len(results["round_trip"]) < 4, results["round_trip"]
        reply, seconds = call(messages)
        results["round_trip"].append({"seconds": seconds, "content": reply["content"][:500], "tool_calls": reply["tool_calls"], "usage": reply["usage"]})
        save()
    assert "4817" in reply["content"], reply

    image = png(320, 240, lambda x, y: (220, 30, 30) if 120 <= x < 200 and 80 <= y < 160 else (30, 60, 200))
    look = {"role": "user", "content": [
        {"type": "text", "text": "What colour is the square in the middle of this image, and what colour is the background? "
                                 "Answer in one short sentence without using tools."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(image).decode()}}]}
    reply, seconds = call([system, look])
    results["image"] = {"seconds": seconds, "content": reply["content"][:500], "usage": reply["usage"]}
    save()
    assert "red" in reply["content"].lower() and "blue" in reply["content"].lower(), reply

    results["efforts"] = {}
    puzzle = {"role": "user", "content": "A tank fills at 3 L/min and drains at 1 L/min, starting with 10 L. When does it hold "
                                         "50 L? Answer with the number of minutes only, without tools."}
    for effort in ("none", "minimal", "low", "medium", "high", "xhigh"):
        reply, seconds = call([puzzle], tools=[], effort=effort, max_tokens=16384)
        results["efforts"][effort] = {"seconds": seconds, "content": reply["content"][:200], "finish_reason": reply["finish_reason"],
                                      "usage": reply["usage"]}
        save()
        assert "20" in reply["content"], (effort, reply)

    essay = [{"role": "user", "content": "Write a detailed 1500-word technical essay about rigid-body contact simulation."}]
    body = {"messages": essay, "max_tokens": 1024, "chat_template_kwargs": {"reasoning_effort": "none"}, "ignore_eos": True}
    reply, seconds = raw(body)
    results["decode_1_stream"] = {"completion_tokens": reply["usage"]["completion_tokens"], "seconds": round(seconds, 2),
                                  "tokens_per_second": round(reply["usage"]["completion_tokens"] / seconds, 1), "timings": reply.get("timings")}
    save()
    with ThreadPoolExecutor(5) as pool:
        started = time.monotonic()
        replies = list(pool.map(lambda _: raw(body), range(5)))
        elapsed = time.monotonic() - started
    tokens = sum(r["usage"]["completion_tokens"] for r, _ in replies)
    results["decode_5_streams"] = {"completion_tokens": tokens, "seconds": round(elapsed, 2), "aggregate_tokens_per_second": round(tokens / elapsed, 1)}
    save()

    filler = " ".join("The quick brown fox jumps over the lazy dog number %d." % i for i in range(9000))
    needle = filler[:len(filler) // 2] + " The secret calibration constant is 7351. " + filler[len(filler) // 2:]
    long = [{"role": "user", "content": needle + "\n\nWhat is the secret calibration constant? Answer with the number only."}]
    for label in ("long_context", "long_context_cached"):
        reply, seconds = raw({"messages": long, "max_tokens": 4096, "chat_template_kwargs": {"reasoning_effort": "low"}})
        content = reply["choices"][0]["message"]["content"] or ""
        results[label] = {"usage": reply["usage"], "seconds": round(seconds, 2), "content": content[:100], "timings": reply.get("timings")}
        save()
        assert "7351" in content, reply
    print(json.dumps(results, indent=1))


if __name__ == "__main__":  # the broker spawns its inference process, which re-imports this module
    main(sys.argv[1], sys.argv[2])
