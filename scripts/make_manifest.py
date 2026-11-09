"""生成发布文件的 SHA256 清单，不包含本机资源、缓存和构建产物。"""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def release_files(root=ROOT):
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if (not path.is_file() or path.is_symlink()
                or any(p in {'.git', '.cache', '__pycache__'} or p.startswith('.venv') for p in relative.parts)
                or relative.parts[0] in {'weights', 'outputs', 'reports'}
                or relative.as_posix().startswith('robot/artifacts/')
                or path.suffix == '.pyc'
                or relative.as_posix() == 'provenance/files.sha256'):
            continue
        yield path


def main():
    lines = []
    for path in release_files():
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        lines.append(f'{digest}  {path.relative_to(ROOT).as_posix()}\n')
    (ROOT / 'provenance/files.sha256').write_text(''.join(lines))
    print(f'已记录 {len(lines)} 个发布文件的 SHA256。')


if __name__ == '__main__':
    main()
