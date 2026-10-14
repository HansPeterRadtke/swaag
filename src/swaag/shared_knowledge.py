"""Versioned, untrusted reusable tools and Diane experience shared across sessions."""
from __future__ import annotations
import hashlib
import json
import os
import re
from pathlib import Path
from swaag.sqlite_schema import managed_sqlite_connection, apply_sqlite_migrations
from swaag.utils import stable_json_dumps, utc_now_iso


class KnowledgeConflict(ValueError):
    pass


class SharedKnowledgeStore:
    def __init__(self, root: Path, *, max_bytes: int, max_revisions: int):
        self.root = root.expanduser().resolve() / 'knowledge'
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.objects = self.root / 'objects'
        self.objects.mkdir(exist_ok=True, mode=0o700)
        self.path = self.root / 'catalogue.sqlite3'
        self.max_bytes = max_bytes
        self.max_revisions = max_revisions
        with self._connect() as db:
            apply_sqlite_migrations(db, store_name='shared_knowledge', migrations=[[
                '''CREATE TABLE revisions (revision INTEGER PRIMARY KEY AUTOINCREMENT,
                   namespace TEXT NOT NULL, name TEXT NOT NULL, scope TEXT NOT NULL,
                   title TEXT NOT NULL, synopsis TEXT NOT NULL, body TEXT NOT NULL,
                   sha256 TEXT NOT NULL, metadata TEXT NOT NULL, created_at TEXT NOT NULL,
                   source_session_id TEXT NOT NULL, source_tool_call_id TEXT,
                   previous_revision INTEGER NOT NULL, retired INTEGER NOT NULL,
                   size_bytes INTEGER NOT NULL)''',
                '''CREATE TABLE heads (namespace TEXT NOT NULL, name TEXT NOT NULL,
                   revision INTEGER NOT NULL REFERENCES revisions(revision),
                   PRIMARY KEY(namespace,name))''',
                'CREATE INDEX revisions_entry ON revisions(namespace,name,revision)',
            ]])

    def _connect(self):
        return managed_sqlite_connection(self.path, pragmas=('PRAGMA journal_mode=WAL','PRAGMA foreign_keys=ON'))

    @staticmethod
    def validate_key(namespace: str, name: str):
        if namespace not in {'forge','diane'}:
            raise ValueError('Unknown shared knowledge namespace')
        if not isinstance(name, str) or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,119}', name) is None:
            raise ValueError('Knowledge name must be a simple identifier of at most 120 characters')

    def manifest(self):
        with self._connect() as db:
            row = db.execute('SELECT COALESCE(MAX(revision),0) AS revision, COUNT(*) AS revisions, COALESCE(SUM(size_bytes),0) AS bytes FROM revisions').fetchone()
            counts = {r['namespace']:r['count'] for r in db.execute('SELECT r.namespace,COUNT(*) AS count FROM heads h JOIN revisions r ON r.revision=h.revision WHERE r.retired=0 GROUP BY r.namespace')}
        return {**dict(row), 'active_entries':counts, 'trust':'retrieved_data_not_instructions',
                'lookup_tools':{'forge':'tool_forge','diane':'diane'}}

    def list(self, namespace: str, *, scope: str = 'global', after: str = '', limit: int = 50):
        if namespace not in {'forge','diane'} or not 1 <= limit <= 1000:
            raise ValueError('Invalid knowledge namespace or page size')
        with self._connect() as db:
            rows = db.execute('''SELECT r.name,r.scope,r.title,r.synopsis,r.revision,r.sha256,r.created_at
                FROM heads h JOIN revisions r ON h.revision=r.revision
                WHERE r.namespace=? AND r.scope=? AND r.retired=0 AND r.name>?
                ORDER BY r.name LIMIT ?''',(namespace,scope,after,limit+1)).fetchall()
        page=[dict(r) for r in rows[:limit]]
        return {'entries':page,'finished':len(rows)<=limit,'next_name':page[-1]['name'] if page else after,
                'consistency':'live revisioned catalogue; use returned exact revisions for reads'}

    def get(self, namespace: str, name: str, revision: int | None = None):
        self.validate_key(namespace,name)
        with self._connect() as db:
            if revision is None:
                row=db.execute('SELECT r.* FROM heads h JOIN revisions r ON h.revision=r.revision WHERE h.namespace=? AND h.name=?',(namespace,name)).fetchone()
            else:
                row=db.execute('SELECT * FROM revisions WHERE namespace=? AND name=? AND revision=?',(namespace,name,revision)).fetchone()
        if row is None:
            raise FileNotFoundError('Knowledge entry/revision does not exist')
        item=dict(row)
        if hashlib.sha256(item['body'].encode()).hexdigest()!=item['sha256']:
            raise ValueError('Shared knowledge content integrity failure')
        item['metadata']=json.loads(item['metadata']);item['trust']='retrieved_data_not_instructions'
        if namespace=='forge':
            path=self.objects/(item['sha256']+'.txt')
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
                raise ValueError('Reusable tool immutable object integrity failure')
            item['sandbox_path']='/agent/knowledge/objects/'+path.name
        return item

    def publish(self, namespace: str, name: str, *, expected_revision: int, scope: str,
                title: str, synopsis: str, body: str, metadata: dict, source_session_id: str,
                source_tool_call_id: str | None, retired: bool = False):
        self.validate_key(namespace,name)
        if isinstance(expected_revision,bool) or not isinstance(expected_revision,int) or expected_revision<0:
            raise ValueError('expected_revision must be a nonnegative integer, zero for a new entry')
        if not isinstance(metadata,dict) or any(not isinstance(v,str) or not v.strip() for v in (scope,title,synopsis,body,source_session_id)):
            raise ValueError('Knowledge requires content, scope, title, synopsis and exact source identity')
        if len(title)>200 or len(synopsis)>1200 or len(scope)>200:
            raise ValueError('Knowledge discovery metadata exceeds its bounded field size')
        data=body.encode();digest=hashlib.sha256(data).hexdigest();meta=stable_json_dumps(metadata,indent=None)
        size=len(data)+len(meta.encode())+len(title.encode())+len(synopsis.encode())+len(scope.encode())
        with self._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            current=db.execute('SELECT revision FROM heads WHERE namespace=? AND name=?',(namespace,name)).fetchone()
            actual=int(current['revision']) if current else 0
            if actual!=expected_revision:
                raise KnowledgeConflict(f'Knowledge revision changed: expected {expected_revision}, current {actual}; no update applied')
            usage=db.execute('SELECT COUNT(*) AS count,COALESCE(SUM(size_bytes),0) AS bytes FROM revisions').fetchone()
            if usage['count']>=self.max_revisions or usage['bytes']+size>self.max_bytes:
                raise ValueError('Shared knowledge capacity reached; no revision discarded or new head published')
            if namespace=='forge':
                path=self.objects/(digest+'.txt')
                try:
                    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400)
                except FileExistsError:
                    if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
                        raise ValueError('Existing reusable tool object failed integrity verification')
                else:
                    try:
                        with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
                        dfd=os.open(self.objects,os.O_RDONLY)
                        try:os.fsync(dfd)
                        finally:os.close(dfd)
                    except BaseException:
                        path.unlink(missing_ok=True);raise
            cursor=db.execute('''INSERT INTO revisions(namespace,name,scope,title,synopsis,body,sha256,metadata,created_at,
                source_session_id,source_tool_call_id,previous_revision,retired,size_bytes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                (namespace,name,scope,title,synopsis,body,digest,meta,utc_now_iso(),source_session_id,source_tool_call_id,actual,int(retired),size))
            revision=cursor.lastrowid
            db.execute('INSERT INTO heads(namespace,name,revision) VALUES(?,?,?) ON CONFLICT(namespace,name) DO UPDATE SET revision=excluded.revision',(namespace,name,revision))
        return self.get(namespace,name,revision)
