"""Import MONTH_selected.npz feature files; does not extract APKs."""
import argparse
import csv
import hashlib
import re
import sys
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True,help='Directory of YYYY-MM_selected.npz files')
    p.add_argument('--output',required=True)
    p.add_argument('--trusted-pickle',action='store_true',help='Only for verified original research artifacts; NumPy pickle can execute code')
    p.add_argument('--deduplicate',action='store_true',help='Keep earliest exact feature vector; log drops')
    p.add_argument('--max-dense-elements',type=int,default=20_000_000)
    a=p.parse_args(); output=Path(a.output)
    if output.exists(): raise FileExistsError(output)
    files=sorted(x for x in Path(a.input).glob('*_selected.npz') if re.fullmatch(r'\d{4}-\d{2}_selected\.npz',x.name))
    if not files: raise ValueError('No monthly NPZ found; combined range files are deliberately excluded')
    output.parent.mkdir(parents=True,exist_ok=True)
    seen=set(); dimension=None; dropped=[]
    temporary=output.with_suffix(output.suffix+'.partial')
    try:
        with temporary.open('w',encoding='utf-8',newline='') as f:
            writer=csv.writer(f)
            for file in files:
                with np.load(file,allow_pickle=a.trusted_pickle) as data:
                    X=data['X_train']; y=np.asarray(data['y_train']).reshape(-1)
                if X.shape==() and X.dtype==object:
                    X=X.item()
                if not hasattr(X,'shape') or len(X.shape)!=2: raise ValueError(f'{file}: expected 2D features')
                if X.shape[0]!=len(y): raise ValueError(f'{file}: mismatched labels')
                if X.shape[0]*X.shape[1]>a.max_dense_elements:
                    raise MemoryError('Dataset too wide/large for this CSV demo adapter. Use a sparse production loader; do not silently densify.')
                if hasattr(X,'toarray'): X=X.toarray()
                X=np.asarray(X,dtype=float)
                if not np.isfinite(X).all() or not np.issubdtype(y.dtype,np.number) or not np.isfinite(y).all():
                    raise ValueError('Need finite numeric features and labels; verify label mapping before import')
                if dimension is None:
                    dimension=X.shape[1]
                    writer.writerow(['sample_id','month','label']+[f'f_{j:05}' for j in range(dimension)])
                if X.shape[1]!=dimension: raise ValueError('Feature spaces differ between months')
                month=file.name[:7]
                for i,(row,label) in enumerate(zip(X,y)):
                    digest=hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()
                    if a.deduplicate and digest in seen:
                        dropped.append((file.name,i,digest));continue
                    seen.add(digest)
                    writer.writerow([f'{file.stem}-row{i}',month,int(label!=0),*row])
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    with output.with_suffix('.dropped.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['source','row','feature_sha256']);w.writerows(dropped)
    print(f'Imported {len(files)} months. Dropped {len(dropped)} exact feature duplicates. Output: {output}')
    print('IDs are source-row IDs, not APK hashes. Family labels are not imported. Zero/nonzero label mapping must be verified.')


if __name__=='__main__': main()
