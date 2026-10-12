"""Durable transport identity for the single authoritative orchestrator.

A started operation with an uncertain outcome is never replayed automatically.
Cancellation is a request, not a claim that a completed external effect was undone.
"""
from __future__ import annotations
import hashlib
import json
import re
import sqlite3
import time
from pathlib import Path
from contextlib import contextmanager

class OperationCanceled(RuntimeError):
    pass

class OrchestratorOperations:
    def __init__(self,root:Path):
        self.path=Path(root)/'orchestrator_operations.sqlite3'
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.connect() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS operations (
                operation_id TEXT PRIMARY KEY, content_sha256 TEXT NOT NULL,
                message TEXT NOT NULL, status TEXT NOT NULL,
                session_id TEXT NOT NULL DEFAULT '', result_json TEXT,
                cancel_requested INTEGER NOT NULL DEFAULT 0,
                created REAL NOT NULL, updated REAL NOT NULL)''')

    @contextmanager
    def connect(self):
        db=sqlite3.connect(self.path,timeout=10)
        db.row_factory=sqlite3.Row
        try:
            db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL')
            with db:yield db
        finally:db.close()

    def claim(self,key:str,message:str):
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,160}',key):raise ValueError('invalid operation_id')
        digest=hashlib.sha256(message.encode()).hexdigest()
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT * FROM operations WHERE operation_id=?',(key,)).fetchone()
            if row:
                if row['content_sha256'] and row['content_sha256']!=digest:raise ValueError('operation_id content conflict')
                return False,self.decode(row)
            now=time.time()
            db.execute('INSERT INTO operations(operation_id,content_sha256,message,status,created,updated) VALUES(?,?,?,?,?,?)',
                       (key,digest,message,'accepted',now,now))
        return True,self.get(key)

    @staticmethod
    def decode(row):
        if row is None:return None
        return {'operation_id':row['operation_id'],'status':row['status'],
                'session_id':row['session_id'],'cancel_requested':bool(row['cancel_requested']),
                'result':json.loads(row['result_json']) if row['result_json'] else None}

    def get(self,key):
        with self.connect() as db:
            return self.decode(db.execute('SELECT * FROM operations WHERE operation_id=?',(key,)).fetchone())

    def activate(self,key,session_id):
        with self.connect() as db:
            db.execute('UPDATE operations SET status=?,session_id=?,updated=? WHERE operation_id=?',
                       ('running',session_id,time.time(),key))
        self.check_cancellation(key)

    def check_cancellation(self,key):
        row=self.get(key)
        if row is None:raise RuntimeError('operation ledger missing')
        if row['cancel_requested']:raise OperationCanceled('orchestrator interruption requested')

    def finish(self,key,status,result=None):
        with self.connect() as db:
            db.execute('UPDATE operations SET status=?,result_json=?,updated=? WHERE operation_id=?',
                       (status,json.dumps(result,ensure_ascii=False) if result is not None else None,time.time(),key))

    def cancel(self,key):
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,160}',key):raise ValueError('invalid operation_id')
        with self.connect() as db:
            now=time.time()
            db.execute("INSERT OR IGNORE INTO operations(operation_id,content_sha256,message,status,cancel_requested,created,updated) VALUES(?,?,?,'canceled',1,?,?)",(key,'','',now,now))
            db.execute("UPDATE operations SET cancel_requested=1,updated=? WHERE operation_id=? AND status IN ('accepted','running')",(time.time(),key))
        return self.get(key)
