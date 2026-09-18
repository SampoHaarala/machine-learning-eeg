"""Download .mat files through the public OSF API; never overwrite raw files.

Usage: python scripts/download_dataset.py --output data/raw --subjects 1 2
Run without --subjects for all participants. The full published data are large.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import urllib.request


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def walk(url):
    while url:
        response = get_json(url)
        for item in response['data']:
            if item['attributes']['kind'] == 'folder':
                child = item['relationships']['files']['links']['related']
                yield from walk(child['href'] if isinstance(child, dict) else child)
            else:
                yield item
        url = response.get('links', {}).get('next')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='data/raw');p.add_argument('--subjects',nargs='*',type=int)
    args=p.parse_args();root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for item in walk('https://api.osf.io/v2/nodes/rsv4z/files/osfstorage/'):
        name=item['attributes']['name']
        match=re.fullmatch(r's_(\d+)\.mat',name,re.I)
        if not match or (args.subjects and int(match[1]) not in args.subjects): continue
        out=root/name
        if out.exists():
            print(f'Already exists; left unchanged: {out}');continue
        partial=root/(name+'.partial')
        if partial.exists():
            raise FileExistsError(f'Review incomplete download first: {partial}')
        url=item['links']['download'];h=hashlib.sha256();count=0
        print(f'Downloading {name}',flush=True)
        with urllib.request.urlopen(url,timeout=120) as response,partial.open('xb') as f:
            while chunk:=response.read(1024*1024):
                f.write(chunk);h.update(chunk);count+=len(chunk)
        expected=item['attributes'].get('size')
        if expected is not None and count!=expected:
            raise ValueError(f'Incomplete {name}: {count}/{expected} bytes')
        # Exclusive creation avoids a rename overwriting another process's file.
        import os
        os.link(partial,out);partial.unlink()
        manifest.append({'file':name,'sha256':h.hexdigest(),'bytes':count,'osf_file_id':item['id']})
        print(f'Completed {name}: {count} bytes',flush=True)
    if not manifest:
        print('No new subject files downloaded. Inspect the OSF repository if its layout changed.')
    else:
        from datetime import datetime,timezone
        name='download_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json'
        (root/name).write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
