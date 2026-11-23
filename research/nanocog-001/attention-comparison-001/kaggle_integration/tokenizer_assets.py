"""Freeze and load a real shared BPE tokenizer for engineering integration.

Uses nanochat's existing GPT-2/tiktoken compatibility path. This is an explicit
50,257-token engineering configuration, not the absent trained 32K tokenizer.
No corpus, checkpoint, model fit, or held-out evaluation is loaded.
"""
import argparse
import base64
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import urllib.request

UPSTREAM_COMMIT = '92d63d4e8bb4df75c3b71618f31ddde2378b2bcd'
UPSTREAM_TOKENIZER_SHA256 = 'e6f12cb88fead3e6d0d45a7dd2af4192280712d92bfde83e2399cc4b3a0860fb'
SOURCE_ASSETS = [
    {'url': 'https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe',
     'sha256': '1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5'},
    {'url': 'https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json',
     'sha256': '196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783'},
]
FIXTURES = [
    'The key moved from Alice to Bob. Who has the key?',
    'def add(a, b):\n    return a + b\n',
    'Lagos, Abuja; 12345. café — 你好世界 🌍',
    '',
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verified_wrapper():
    import nanochat.tokenizer as module
    if digest(module.__file__) != UPSTREAM_TOKENIZER_SHA256:
        raise RuntimeError('Imported nanochat tokenizer source does not match the pin')
    return module.RustBPETokenizer


def prepare(destination):
    import tiktoken
    RustBPETokenizer = verified_wrapper()
    if importlib.metadata.version('tiktoken') != '0.11.0':
        raise RuntimeError('Tokenizer preparation requires pinned tiktoken 0.11.0')
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    # Observe and verify the actual raw bytes, independently of tiktoken's cache.
    cache = destination / 'verified_raw_assets'
    cache.mkdir(exist_ok=True)
    observed_assets = []
    for asset in SOURCE_ASSETS:
        with urllib.request.urlopen(asset['url'], timeout=30) as response:
            raw = response.read(4_000_001)
        if len(raw) > 4_000_000:
            raise RuntimeError('Tokenizer source exceeds the declared byte cap')
        observed = hashlib.sha256(raw).hexdigest()
        if observed != asset['sha256']:
            raise RuntimeError('Tokenizer raw source SHA-256 mismatch')
        cache_name = hashlib.sha1(asset['url'].encode()).hexdigest()
        (cache / cache_name).write_bytes(raw)
        observed_assets.append({'url': asset['url'], 'expected_sha256': asset['sha256'],
                                'observed_sha256': observed,
                                'size_bytes': len(raw)})
    previous_cache = os.environ.get('TIKTOKEN_CACHE_DIR')
    os.environ['TIKTOKEN_CACHE_DIR'] = str(cache.resolve())
    # Also check a potentially pre-existing in-process tiktoken encoding against
    # a fresh constructor reading only the independently verified raw bytes.
    try:
        tokenizer = RustBPETokenizer.from_pretrained('gpt2')
        encoding = tokenizer.enc
        if (encoding.name != 'gpt2' or tokenizer.get_vocab_size() != 50257
                or tokenizer.get_bos_token_id() != 50256):
            raise RuntimeError('Unexpected tokenizer configuration')
        from tiktoken_ext.openai_public import gpt2
        fresh = gpt2()
        if (encoding._mergeable_ranks != fresh['mergeable_ranks']
                or encoding._special_tokens != fresh['special_tokens']
                or encoding._pat_str != fresh['pat_str']):
            raise RuntimeError('Tokenizer encoding differs from verified source assets')
    finally:
        if previous_cache is None:
            os.environ.pop('TIKTOKEN_CACHE_DIR', None)
        else:
            os.environ['TIKTOKEN_CACHE_DIR'] = previous_cache
    spec = {
        'name': encoding.name,
        'pat_str': encoding._pat_str,
        'explicit_n_vocab': encoding.n_vocab,
        'mergeable_ranks': [[base64.b64encode(token).decode('ascii'), rank]
                            for token, rank in sorted(encoding._mergeable_ranks.items(),
                                                      key=lambda item: item[1])],
        'special_tokens': encoding._special_tokens,
        'bos_token': '<|endoftext|>',
    }
    encoding_path = destination / 'encoding.json'
    encoding_path.write_text(json.dumps(spec, sort_keys=True, separators=(',', ':')) + '\n')
    checks = []
    for text in FIXTURES:
        tokens = tokenizer.encode(text)
        if tokenizer.decode(tokens) != text:
            raise RuntimeError('Tokenizer fixture did not round-trip')
        checks.append({'utf8_sha256': hashlib.sha256(text.encode()).hexdigest(),
                       'token_ids': tokens, 'round_trip': True})
    manifest = {
        'schema': 'cog.shared-tokenizer.v1',
        'scope': 'engineering_integration_only',
        'nanochat_commit': UPSTREAM_COMMIT,
        'imported_nanochat_tokenizer_sha256': UPSTREAM_TOKENIZER_SHA256,
        'tiktoken_version': importlib.metadata.version('tiktoken'),
        'rustbpe_version': importlib.metadata.version('rustbpe'),
        'source_assets': observed_assets,
        'encoding_sha256': digest(encoding_path),
        'vocab_size': tokenizer.get_vocab_size(),
        'bos_token_id': tokenizer.get_bos_token_id(),
        'fixtures': checks,
        'training_or_evaluation_corpus': None,
        'trained_32k_nanochat_tokenizer_reproduced': False,
    }
    (destination / 'tokenizer_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def load(destination):
    import tiktoken
    RustBPETokenizer = verified_wrapper()
    destination = Path(destination)
    manifest = json.loads((destination / 'tokenizer_manifest.json').read_text())
    if digest(destination / 'encoding.json') != manifest['encoding_sha256']:
        raise RuntimeError('Frozen tokenizer content mismatch')
    if importlib.metadata.version('tiktoken') != manifest['tiktoken_version']:
        raise RuntimeError('Tokenizer implementation version mismatch')
    spec = json.loads((destination / 'encoding.json').read_text())
    ranks = {base64.b64decode(token, validate=True): rank
             for token, rank in spec['mergeable_ranks']}
    enc = tiktoken.Encoding(name=spec['name'], pat_str=spec['pat_str'],
                           explicit_n_vocab=spec['explicit_n_vocab'],
                           mergeable_ranks=ranks, special_tokens=spec['special_tokens'])
    tokenizer = RustBPETokenizer(enc, spec['bos_token'])
    if tokenizer.get_vocab_size() != manifest['vocab_size']:
        raise RuntimeError('Tokenizer vocabulary mismatch')
    if tokenizer.get_bos_token_id() != manifest['bos_token_id']:
        raise RuntimeError('Tokenizer BOS mismatch')
    for text, check in zip(FIXTURES, manifest['fixtures'], strict=True):
        ids = tokenizer.encode(text)
        if ids != check['token_ids'] or tokenizer.decode(ids) != text:
            raise RuntimeError('Frozen tokenizer behavior mismatch')
    return tokenizer, manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('destination')
    args = parser.parse_args()
    print(json.dumps(prepare(args.destination), indent=2))
