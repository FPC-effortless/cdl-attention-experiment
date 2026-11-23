"""Build one credential-free Kaggle script from explicit, pinned source inputs."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import stat
import subprocess
import zipfile

NANOCHAT = '92d63d4e8bb4df75c3b71618f31ddde2378b2bcd'
CANDIDATE = '1a91168b216a49543cc839153fa299a262320730'
CANDIDATE_EQUIVALENT = '39be434ff160d4e6507df8d2114979ba5529caff'
LOCAL_FILES = ('adapters.py', 'test_adapters.py', 'tokenizer_assets.py',
               'runner_core.py', 'bootstrap-requirements.txt', 'RUN_PROTOCOL.md')

BOOTSTRAP = r'''# Source-pinned private engineering job. No credential is embedded.
import base64, hashlib, importlib, json, os
from pathlib import Path, PurePosixPath
import runpy, stat, subprocess, sys, tempfile, time, traceback, zipfile, io

MANIFEST = json.loads(__COG_MANIFEST_JSON__)
ARCHIVE_SHA256 = '__COG_ARCHIVE_SHA256__'
PAYLOAD = '__COG_PAYLOAD_B64__'
started = time.monotonic()
record = {'status': 'BOOTSTRAPPING', 'archive_sha256': ARCHIVE_SHA256,
          'source_manifest': MANIFEST, 'credential_included': False,
          'optimizer_updates': 0, 'scientific_comparison_admitted': False}
Path('run_source_manifest.json').write_text(json.dumps(MANIFEST, indent=2)+'\n')
try:
    raw = base64.b64decode(PAYLOAD, validate=True)
    if hashlib.sha256(raw).hexdigest() != ARCHIVE_SHA256:
        raise RuntimeError('Source archive checksum mismatch')
    source = Path(tempfile.mkdtemp(prefix='cog_source_'))
    expected = {item['bundle_path']: item for item in MANIFEST['files']}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = archive.infolist()
        if len(members) != len(expected) or {item.filename for item in members} != set(expected):
            raise RuntimeError('Archive member set differs from reviewed allowlist')
        if sum(item.file_size for item in members) > 10_000_000:
            raise RuntimeError('Source archive exceeds uncompressed size cap')
        for item in members:
            name = PurePosixPath(item.filename)
            mode = item.external_attr >> 16
            if name.is_absolute() or '..' in name.parts or not stat.S_ISREG(mode):
                raise RuntimeError('Nonregular/unsafe source archive member')
            content = archive.read(item)
            spec = expected[item.filename]
            if len(content) != spec['size_bytes'] or hashlib.sha256(content).hexdigest() != spec['sha256']:
                raise RuntimeError('Source member checksum mismatch: '+item.filename)
            destination = source.joinpath(*name.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
    target = Path(tempfile.mkdtemp(prefix='cog_deps_'))
    requirement = source/'integration/bootstrap-requirements.txt'
    dependency_report = Path.cwd()/'dependency_install.json'
    # Install only two explicitly version/hash-pinned binary wheels. The base
    # image supplies Torch and transitive dependencies, whose versions are recorded.
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
                    '--index-url', 'https://pypi.org/simple', '--no-deps',
                    '--only-binary=:all:', '--require-hashes', '--target', str(target),
                    '--report', str(dependency_report), '-r', str(requirement)],
                   check=True, timeout=120)
    sys.path[:0] = [str(target), str(source/'upstream'), str(source/'candidate'),
                   str(source/'integration')]
    importlib.invalidate_caches()
    record['status'] = 'BOOTSTRAP_VERIFIED'
    record['elapsed_seconds'] = time.monotonic()-started
    Path('bootstrap_result.json').write_text(json.dumps(record, indent=2)+'\n')
    namespace = runpy.run_path(str(source/'integration/runner_core.py'), run_name='cog_worker')
    exit_code = namespace['main']()
except Exception as exc:
    record['status'] = 'FAILED_BOOTSTRAP'
    record['error'] = {'type': type(exc).__name__, 'message': str(exc),
                       'traceback': traceback.format_exc()}
    record['elapsed_seconds'] = time.monotonic()-started
    Path('bootstrap_result.json').write_text(json.dumps(record, indent=2)+'\n')
    raise
raise SystemExit(exit_code)
'''


def git_bytes(repo, revision, path):
    return subprocess.check_output(['git', '-C', str(repo), 'show', revision + ':' + path])


def git_paths(repo, revision, path):
    return subprocess.check_output(['git', '-C', str(repo), 'ls-tree', '-r', '--name-only',
                                    revision, path], text=True).splitlines()


def build(args):
    local = Path(__file__).resolve().parent
    files, payload = [], {}

    def add(bundle_path, content, provenance):
        if bundle_path in payload:
            raise ValueError('Duplicate source member')
        payload[bundle_path] = content
        files.append({'bundle_path': bundle_path, 'size_bytes': len(content),
                      'sha256': hashlib.sha256(content).hexdigest(), **provenance})

    for path in git_paths(args.nanochat, NANOCHAT, 'nanochat'):
        if path.count('/') == 1 and path.endswith('.py'):
            add('upstream/' + path, git_bytes(args.nanochat, NANOCHAT, path),
                {'kind': 'upstream_source', 'repository': 'karpathy/nanochat',
                 'commit': NANOCHAT, 'path': path})
    add('upstream/LICENSE', git_bytes(args.nanochat, NANOCHAT, 'LICENSE'),
        {'kind': 'license', 'repository': 'karpathy/nanochat', 'commit': NANOCHAT})
    prefix = 'research/nanocog-001/'
    for path in git_paths(args.candidate, CANDIDATE, prefix + 'nanocog'):
        if path.endswith('.py') and not Path(path).name.startswith('test_'):
            content = git_bytes(args.candidate, CANDIDATE, path)
            if content != git_bytes(args.candidate, CANDIDATE_EQUIVALENT, path):
                raise ValueError('Candidate package differs between declared reference commits')
            add('candidate/' + path.removeprefix(prefix), content,
                {'kind': 'candidate_source', 'repository': 'FPC-effortless/cdl-attention-experiment',
                 'commit': CANDIDATE, 'equivalent_reference_commit': CANDIDATE_EQUIVALENT,
                 'path': path})
    add('candidate/NOTICE_NANOCHAT_LICENSE',
        git_bytes(args.candidate, CANDIDATE, prefix + 'NOTICE_NANOCHAT_LICENSE'),
        {'kind': 'license', 'repository': 'FPC-effortless/cdl-attention-experiment',
         'commit': CANDIDATE})
    for name in LOCAL_FILES:
        add('integration/' + name, (local / name).read_bytes(),
            {'kind': 'reviewed_integration_source', 'path': name})
    for name in ('encoding.json', 'tokenizer_manifest.json', 'GPT2_LICENSE.txt'):
        add('tokenizer/' + name, (Path(args.tokenizer) / name).read_bytes(),
            {'kind': 'verified_tokenizer_asset', 'path': name})
    files.sort(key=lambda item: item['bundle_path'])
    manifest = {
        'schema': 'cog.private-kaggle-source-bundle.v1',
        'run_id': 'COG-ATTENTION-KAGGLE-ADAPTER-001',
        'source_only_allowlist': True, 'credential_included': False,
        'core_timeout_seconds': 720, 'requested_server_timeout_seconds': 900,
        'requested_accelerator': 'NvidiaTeslaT4', 'is_private': True,
        'worker_internet': 'enabled only for hash-pinned PyPI binary bootstrap; no model/data downloads',
        'files': files,
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(payload):
            info = zipfile.ZipInfo(path, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, payload[path])
    raw = buffer.getvalue()
    archive_hash = hashlib.sha256(raw).hexdigest()
    script = BOOTSTRAP.replace('__COG_MANIFEST_JSON__', repr(json.dumps(manifest, sort_keys=True)))
    script = script.replace('__COG_ARCHIVE_SHA256__', archive_hash)
    script = script.replace('__COG_PAYLOAD_B64__', base64.b64encode(raw).decode('ascii'))
    destination = Path(args.destination)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'kernel.py').write_text(script)
    metadata = {'id': args.owner + '/cog-attention-adapter-001',
                'title': 'COG Attention Adapter 001', 'code_file': 'kernel.py',
                'language': 'python', 'kernel_type': 'script', 'is_private': True,
                'enable_gpu': True, 'enable_tpu': False, 'enable_internet': True,
                'dataset_sources': [], 'kernel_sources': [], 'competition_sources': [],
                'model_sources': []}
    (destination / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    receipt = {'archive_sha256': archive_hash,
               'kernel_source_sha256': hashlib.sha256(script.encode()).hexdigest(),
               'kernel_source_bytes': len(script.encode()), 'metadata': metadata,
               'source_manifest': manifest}
    (local / 'BUNDLE_RECEIPT.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({key: receipt[key] for key in
                      ('archive_sha256', 'kernel_source_sha256', 'kernel_source_bytes')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nanochat', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--destination', required=True)
    parser.add_argument('--owner', required=True)
    build(parser.parse_args())
