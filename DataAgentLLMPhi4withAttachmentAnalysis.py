# DataAgentLLMPhi4withAttachmentAnalysis.py
# Phi-4 Free-Style Multi-CSV Analyst:
# - Any CSV(s) -> SQLite (streamed)
# - LLM writes tiny Python that calls `tools.*` (no file access)
# - Helpers for dates/month grouping; robust header resolution
# - Safe sandbox with needed builtins (print/next/ValueError/…)
# - DB reused across follow-ups; detailed logging

import os, re, json, textwrap, tempfile, sqlite3, queue, inspect, logging, hashlib, time
from typing import Any, Dict, List, Optional, Tuple
from multiprocessing import Process, Queue as MPQueue

import gradio as gr
import pandas as pd
from gpt4all import GPT4All

# ---------------- Logging ----------------
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
ts = time.strftime("%Y%m%d_%H%M%S")
LOG_PATH = os.path.join(LOG_DIR, f"phi4_agent_{ts}.log")

logging.basicConfig(
    filename=LOG_PATH,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
print(f"[INFO] Logs -> {LOG_DIR}")

# ---------------- Runtime / Model ----------------
_CPU = max(2, (os.cpu_count() or 8) - 1)
for var in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(var, str(_CPU))

MODEL_NAME = "phi-4-Q4_K_S.gguf"
MODEL_DIR  = r"C:\Users\balum\OneDrive\Documents\AI\LLMs\models"
USE_GPU = False

RUNTIME_CFG = dict(model_path=MODEL_DIR, allow_download=False, n_ctx=8192, mlock=True, mmap=False)
CPU_INIT = dict(device="cpu", n_gpu_layers=0, n_batch=1536, n_threads=_CPU)
GPU_INIT = dict(device="gpu", n_gpu_layers=-1, n_batch=1536)
GEN_CFG = dict(max_tokens=700, temp=0.15, top_p=0.9, repeat_penalty=1.07)
STOP_MARKERS = ["### User:", "### System:", "<|end|>"]

SYSTEM_PLANNER = (
    "You are a free-form data analyst. You cannot read files directly; use `tools` only.\n"
    "Steps you MUST follow:\n"
    "  1) tables = tools.tables(); pick a table\n"
    "  2) cols   = tools.columns(table)\n"
    "  3) If you need dates/months: dcols = tools.find_date_columns(table)\n"
    "  4) For monthly counts: result = tools.groupby_month_counts(table, date_col=None, year=None)\n"
    "  5) Otherwise use tools.sql(SELECT ..) or tools.top_counts(table, column)\n"
    "Rules:\n"
    "  • Assign a variable named `result` (<=50 rows)\n"
    "  • Only Python that calls `tools.*`. No file I/O, OS, or network.\n"
    "Return ONLY a Python fenced block. Do not return prose."
)

SYSTEM_EXPLAIN = (
    "You are a concise analyst. Using ONLY the result JSON and the user question,\n"
    "explain the answer in under 180 words. Include exact counts/percentages and any caveats."
)

def _supported_kwargs():
    try:
        params = inspect.signature(GPT4All.__init__).parameters
        s = set(params); s.discard("self"); return s
    except Exception:
        return None

_SUPPORTED = _supported_kwargs()

def _filter_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    if not _SUPPORTED:
        return kwargs
    return {k:v for k,v in kwargs.items() if k in _SUPPORTED}

def _init_model():
    try:
        base = dict(RUNTIME_CFG)
        if USE_GPU:
            cfg = _filter_kwargs({**base, **GPU_INIT})
            logging.info(f"[MODEL] GPU init: {cfg}")
            return GPT4All(MODEL_NAME, **cfg)
        cfg = _filter_kwargs({**base, **CPU_INIT})
        logging.info(f"[MODEL] CPU init: {cfg}")
        return GPT4All(MODEL_NAME, **cfg)
    except Exception as e:
        logging.exception("[MODEL] Init failed; raising:")
        raise

model = _init_model()

# --------------- CSV -> SQLite Tools ----------------

def _sanitize_table_name(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    base = re.sub(r"[^A-Za-z0-9_]+", "_", base)
    if not re.match(r"^[A-Za-z_]", base):
        base = "t_" + base
    return base[:60]

def _normalize_col(col: str) -> str:
    c = re.sub(r"[^A-Za-z0-9_]+", "_", str(col)).strip("_")
    return c or "col"

LIKELY_DATE_PAT = re.compile(r"(date|time|occ|reported|rpt|dt|year|month)", re.I)

class SqliteTools:
    """
    High-level API exposed to the LLM.
    Use help(): tools.help()
    """
    _ACTIVE_DB: Optional[str] = None  # path for reuse across follow-ups

    def __init__(self, csv_paths: List[str], chunksize: int = 250_000):
        self.csv_paths = csv_paths
        self.chunksize = chunksize
        # Stable DB path per run/session (hash of file list) so we can reuse for follow-ups.
        sig = hashlib.md5(("||".join(csv_paths)).encode()).hexdigest()[:16]
        self.db_path = SqliteTools._ACTIVE_DB or os.path.join(tempfile.gettempdir(), f"phi4_sqlite_{os.getpid()}_{sig}.db")
        self.con: Optional[sqlite3.Connection] = None
        self._table_names: List[str] = []
        self._name_map: Dict[str, Dict[str, str]] = {}  # table -> {original->normalized}
        self._rev_map: Dict[str, Dict[str, str]] = {}   # table -> {normalized->original}

        fresh = not os.path.exists(self.db_path)
        self.con = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        self.con.execute("PRAGMA journal_mode=WAL;")
        self.con.execute("PRAGMA synchronous=OFF;")

        if fresh:
            logging.info(f"[INGEST] Creating DB {self.db_path}")
            self._ingest_all()
        else:
            logging.info(f"[TOOLS] Reattached DB: {self.db_path}")
            self._table_names = [r[0] for r in self.con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY 1"
            ).fetchall()]

        SqliteTools._ACTIVE_DB = self.db_path
        logging.info(f"[INGEST] Available tables: {self._table_names}")

    # For Windows fork-safety
    def __getstate__(self):
        d = self.__dict__.copy()
        d['con'] = None
        return d
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.con = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        self.con.execute("PRAGMA journal_mode=WAL;")
        self.con.execute("PRAGMA synchronous=OFF;")

    # ---------- Ingestion ----------
    def _ingest_all(self):
        for p in self.csv_paths:
            tbl = _sanitize_table_name(p)
            self._csv_to_table(p, tbl)
            self._table_names.append(tbl)

    def _csv_to_table(self, path: str, table: str):
        logging.info(f"[INGEST] CSV -> table: {table}")
        # Sample header & dtypes
        sample = pd.read_csv(path, nrows=3000, low_memory=False)
        # Build maps
        name_map = {}
        norm_cols = []
        for c in sample.columns:
            nc = _normalize_col(c)
            name_map[str(c)] = nc
            norm_cols.append(nc)
        self._name_map[table] = name_map
        self._rev_map[table]  = {v:k for k,v in name_map.items()}
        sample.columns = norm_cols

        # Parse likely date columns to datetime so strftime() works
        for c in list(sample.columns):
            if LIKELY_DATE_PAT.search(c):
                try:
                    sample[c] = pd.to_datetime(sample[c], errors="coerce", infer_datetime_format=True)
                except Exception:
                    pass

        # Create empty table
        sample.head(0).to_sql(table, self.con, index=False, if_exists='replace')

        # Append sample rows
        sample.to_sql(table, self.con, index=False, if_exists='append')

        # Stream the rest
        reader = pd.read_csv(path, chunksize=self.chunksize, skiprows=range(1, 3001), header=0, low_memory=False, names=list(sample.columns))
        for chunk in reader:
            # parse dates again for same cols
            for c in list(chunk.columns):
                if LIKELY_DATE_PAT.search(c):
                    try:
                        chunk[c] = pd.to_datetime(chunk[c], errors="coerce", infer_datetime_format=True)
                    except Exception:
                        pass
            chunk.to_sql(table, self.con, index=False, if_exists='append')

        # Minimal indexing hints
        for col in sample.columns:
            if re.search(r"(id|code|zip|date|time)$", col, re.I):
                try:
                    self.con.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_{col} ON {table}({col});")
                except Exception:
                    pass
        self.con.commit()

    # ---------- Public API (used by LLM code) ----------
    def help(self) -> str:
        return textwrap.dedent("""
        tools API:
          • tables() -> List[str]
          • columns(table) -> List[str]                      (normalized SQL-safe names)
          • columns_info(table) -> List[{'name','type'}]     (detailed schema)
          • sample(table, n=20) -> List[RowDict]
          • profile(table, top_n=10) -> lightweight value counts per column
          • resolve_col(table, name_like) -> normalized column (case/space-insensitive)
          • find_date_columns(table) -> List[str]            (likely date/time columns)
          • groupby_month_counts(table, date_col=None, year=None, top=12) -> [{month, count}]
          • top_counts(table, column, top=10) -> [{value, count}]
          • sql(query, limit=1000) -> List[RowDict]          (adds LIMIT if missing; SELECT-only)
        Notes:
          - Column names shown by columns() are normalized (spaces→underscores). resolve_col() maps
            names like "DATE OCC" → "DATE_OCC".
          - Use strftime in SQL: strftime('%Y', DATE_OCC)='2020', month=strftime('%m', DATE_OCC)
        """)

    def tables(self) -> List[str]:
        return list(self._table_names)

    def columns(self, table: str) -> List[str]:
        return [r[1] for r in self.con.execute(f"PRAGMA table_info({table})").fetchall()]

    def columns_info(self, table: str) -> List[Dict[str, Any]]:
        cur = self.con.execute(f"PRAGMA table_info({table});")
        return [{"name": r[1], "type": r[2]} for r in cur.fetchall()]

    def resolve_col(self, table: str, name_like: str) -> Optional[str]:
        """Map fuzzy user/LLM header to normalized column."""
        cand = name_like.strip().lower().replace(" ", "_")
        cols = set(self.columns(table))
        # direct match
        if name_like in cols: return name_like
        if cand in cols: return cand
        # try name map (original -> normalized)
        nm = self._name_map.get(table, {})
        for orig, norm in nm.items():
            if orig.lower() == name_like.lower(): return norm
            if orig.lower().replace(" ", "_") == cand: return norm
        # partial contains
        for c in cols:
            if cand in c.lower(): return c
        return None

    def sample(self, table: str, n: int = 20) -> List[Dict[str, Any]]:
        df = pd.read_sql_query(f"SELECT * FROM {table} LIMIT {int(n)};", self.con)
        return df.to_dict(orient='records')

    def profile(self, table: str, top_n: int = 10) -> Dict[str, Any]:
        cols = self.columns(table)
        nrows = self.con.execute(f"SELECT COUNT(1) FROM {table}").fetchone()[0]
        prof = {"table": table, "rows": int(nrows), "columns": []}
        for c in cols:
            try:
                top = pd.read_sql_query(
                    f"SELECT {c} AS val, COUNT(*) AS cnt FROM {table} GROUP BY {c} ORDER BY cnt DESC LIMIT {top_n}",
                    self.con
                ).to_dict(orient='records')
            except Exception:
                top = []
            try:
                nulls = self.con.execute(f"SELECT COUNT(1) FROM {table} WHERE {c} IS NULL").fetchone()[0]
            except Exception:
                nulls = 0
            prof["columns"].append({"name": c, "nulls": int(nulls), "top": top})
        return prof

    def top_counts(self, table: str, column: str, top: int = 10) -> List[Dict[str, Any]]:
        col = self.resolve_col(table, column) or column
        q = f"SELECT {col} AS value, COUNT(*) AS count FROM {table} GROUP BY {col} ORDER BY count DESC LIMIT {int(top)}"
        logging.info(f"[TOOLS] top_counts({table}.{col}, {top})")
        return pd.read_sql_query(q, self.con).to_dict(orient='records')

    def find_date_columns(self, table: str) -> List[str]:
        return [c for c in self.columns(table) if LIKELY_DATE_PAT.search(c)]

    def groupby_month_counts(self, table: str, date_col: Optional[str]=None, year: Optional[int]=None, top: int = 12) -> List[Dict[str, Any]]:
        if date_col:
            dc = self.resolve_col(table, date_col) or date_col
        else:
            dcols = self.find_date_columns(table)
            if not dcols:
                return []
            dc = dcols[0]
        where = ""
        if year is not None:
            where = f"WHERE CAST(strftime('%Y', {dc}) AS INT) = {int(year)}"
        q = f"""
        SELECT CAST(strftime('%m', {dc}) AS INT) AS month, COUNT(*) AS crime_count
        FROM {table}
        {where}
        GROUP BY month
        ORDER BY crime_count DESC, month ASC
        LIMIT {int(top)}
        """
        logging.info(f"[TOOLS] month_counts({table}.{dc}, year={year})")
        return pd.read_sql_query(q, self.con).to_dict(orient='records')

    def _clean_sql(self, query: str, limit: int) -> str:
        q = query.strip().strip(";")
        if not re.match(r"(?is)^select\s", q):
            raise ValueError("Only SELECT queries are allowed")
        if " from " not in q.lower():
            # Heuristic: inject FROM first table if user forgot
            tbl = self._table_names[0] if self._table_names else ""
            q = re.sub(r"(?is)^select\s+", f"SELECT ", q) + f" FROM {tbl}"
        if " limit " not in q.lower():
            q += f" LIMIT {int(limit)}"
        return q

    def sql(self, query: str, limit: int = 1000) -> List[Dict[str, Any]]:
        q = self._clean_sql(query, limit)
        logging.info(f"[TOOLS] SQL (clean) -> {q}")
        df = pd.read_sql_query(q, self.con)
        if len(df) > 500: df = df.head(500)
        return df.to_dict(orient='records')

# --------------- Sandbox ----------------
class CodeSandbox:
    def __init__(self, timeout_sec: int = 60): self.timeout = timeout_sec

    def _worker(self, code: str, tools: SqliteTools, q: MPQueue):
        safe_builtins = {
            # constants
            'True': True, 'False': False, 'None': None,
            # types / utils
            'len': len, 'range': range, 'enumerate': enumerate, 'zip': zip,
            'min': min, 'max': max, 'sum': sum, 'abs': abs, 'sorted': sorted,
            'list': list, 'dict': dict, 'set': set, 'tuple': tuple, 'str': str,
            'int': int, 'float': float, 'bool': bool, 'any': any, 'all': all,
            'isinstance': isinstance, 'print': print, 'next': next, 'ValueError': ValueError
        }
        glob = {'__builtins__': safe_builtins, 'tools': tools}
        loc: Dict[str, Any] = {}
        try:
            exec(code, glob, loc)
            res = loc.get('result', glob.get('result'))
            # jsonable
            def _jsonable(x):
                if isinstance(x, (list, dict, str, int, float, bool)) or x is None: return x
                return json.loads(json.dumps(x, default=str))
            q.put({"ok": True, "result": _jsonable(res)})
        except Exception as e:
            logging.exception("[SANDBOX] Error during execution")
            q.put({"ok": False, "error": str(e)})

    def run(self, code: str, tools: SqliteTools) -> Dict[str, Any]:
        logging.info("[SANDBOX] Forking worker...")
        q = MPQueue()
        p = Process(target=self._worker, args=(code, tools, q))
        p.start()
        p.join(self.timeout)
        if p.is_alive():
            logging.error("[SANDBOX] Execution timed out.")
            p.terminate(); p.join()
            return {"ok": False, "error": f"Execution timed out after {self.timeout}s"}
        try:
            out = q.get_nowait()
            logging.info(f"[SANDBOX] Outcome: {out if out.get('ok') else 'ERROR'}")
            return out
        except queue.Empty:
            logging.error("[SANDBOX] No result returned")
            return {"ok": False, "error": "No result returned"}

sandbox = CodeSandbox(timeout_sec=60)

# --------------- Agent plumbing ----------------
def _messages_to_prompt(messages, system):
    parts = [f"### System\n{system}\n"]
    for m in messages:
        parts.append(f"### {m['role'].capitalize()}\n{m['content']}\n")
    parts.append("### Assistant\n")
    return "\n".join(parts)

def _gen(prompt: str) -> str:
    out = ""
    for tok in model.generate(prompt, streaming=False, **GEN_CFG):
        out += tok
        if any(out.endswith(m) or m in out[-64:] for m in STOP_MARKERS): break
    return out.strip()

INSTRUCTIONS_TOOLS = textwrap.dedent("""
Write ONLY Python that calls `tools` and assigns `result`. Examples:

# See schema
tables = tools.tables()
table = tables[0]
cols = tools.columns(table)

# Month counts (auto date column; optional year)
result = tools.groupby_month_counts(table, year=2020)

# Month counts using a specific column
dcols = tools.find_date_columns(table)
result = tools.groupby_month_counts(table, date_col=dcols[0], year=2020)

# Top values
result = tools.top_counts(table, 'Crm_Cd_Desc', top=10)

If you choose tools.sql(), include a SELECT and small LIMIT. Do NOT return prose.
""")

def ask_phi_for_code(user_task: str) -> str:
    prompt = _messages_to_prompt(
        [{"role":"user","content": f"Task: {user_task}\n\n{INSTRUCTIONS_TOOLS}"}],
        SYSTEM_PLANNER
    )
    raw = _gen(prompt)
    code = raw
    if "```" in raw:
        segs = raw.split("```", 2)
        if len(segs) >= 2:
            block = segs[1]
            if block.lower().startswith("python"): block = block[6:].lstrip("\n")
            code = block
    code = code.strip()
    # Second pass if missing essentials
    if ("tools." not in code) or ("result" not in code):
        strict = _messages_to_prompt(
            [{"role":"user","content": f"RETURN ONLY PYTHON.\nTask: {user_task}\n\n{INSTRUCTIONS_TOOLS}"}],
            SYSTEM_PLANNER
        )
        raw2 = _gen(strict)
        if "```" in raw2:
            segs2 = raw2.split("```", 2)
            if len(segs2) >= 2:
                block2 = segs2[1]
                if block2.lower().startswith("python"): block2 = block2[6:].lstrip("\n")
                code2 = block2.strip()
                if "tools." in code2 and "result" in code2:
                    code = code2
    logging.info("[MODEL CODE]\n" + code)
    return code

def explain_with_phi(result_json: Any, user_q: str) -> str:
    payload = json.dumps(result_json)[:28000]
    prompt = _messages_to_prompt(
        [{"role":"user","content": f"Question: {user_q}\nResult JSON (truncated):\n{payload}"}],
        SYSTEM_EXPLAIN
    )
    return _gen(prompt)

# --------------- UI + Flow ----------------
file_state = gr.State([])         # stores file paths
tools_state = gr.State(None)      # stores a live SqliteTools for reuse

def chat_fn(message, history, files):
    logging.info(f"[CHAT] Message: {message}")
    logging.info(f"[CHAT] Files: {files}")
    paths = files or file_state.value
    if files: file_state.value = files

    # Reuse SqliteTools/DB if possible
    tools = tools_state.value
    if tools is None or not isinstance(tools, SqliteTools):
        if not paths:
            return "Attach one or more CSV files to begin."
        tools = SqliteTools(paths)
        tools_state.value = tools
        logging.info(f"[CHAT] Created new SqliteTools DB: {tools.db_path}")
    else:
        logging.info(f"[CHAT] Reusing existing SqliteTools and DB: {tools.db_path}")

    # If user sends empty/irrelevant message, return overview
    if not message or not str(message).strip():
        try:
            tbls = tools.tables()
            if not tbls:
                return "No tables available."
            table = tbls[0]
            prof = tools.profile(table, top_n=5)
            return "No specific question detected. Here's a quick overview:\n\n" + json.dumps(prof, indent=2)[:1400]
        except Exception as e:
            logging.exception("[CHAT] Overview failed")
            return f"Could not create overview: {e}"

    code = ask_phi_for_code(message)
    if not code:
        return "Could not produce code for this task. Try rephrasing."

    outcome = sandbox.run(code, tools)
    if not outcome.get("ok"):
        # One retry: if the model forgot a SELECT or used bad SQL, try to answer via helpers if we can detect intent
        err = outcome.get("error","")
        logging.warning(f"[CHAT] First attempt failed: {err}")
        # Fallback for “months of highest crime” intents
        if re.search(r"month|highest crime|DATE|date", message, re.I):
            try:
                tbl = tools.tables()[0]
                rows = tools.groupby_month_counts(tbl)
                if not rows:
                    rows = tools.groupby_month_counts(tbl, year=2020)
                if rows:
                    ans = explain_with_phi(rows, message)
                    return ans + "\n\nRaw result (truncated):\n" + json.dumps(rows, indent=2)[:1200]
            except Exception:
                pass
        return f"Code error: {err}\n\nGenerated code:\n{code}"

    result = outcome.get("result", [])
    if not result:
        logging.warning("[CHAT] Empty result")
        return "No matching rows were found or the query returned nothing. Try being more specific (filters, columns, or year)."

    answer = explain_with_phi(result, message)
    preview = "\n\nRaw result (truncated):\n" + json.dumps(result, indent=2)[:1200]
    logging.info(f"[CHAT] Result preview: {json.dumps(result)[:200]}")
    return answer + preview

with gr.Blocks(title="Phi-4 Free-Style Multi-File Analyst") as ui:
    gr.Markdown("# Phi-4 Free-Style Analyst (Multi-CSV, Joins, SQL via Tools)\nUpload one or more CSVs. The model discovers tables/columns, plans joins, queries via tools.sql or helpers, and explains results.")
    files_input = gr.Files(label="Attach CSV files", file_types=[".csv"], type="filepath")
    chat = gr.ChatInterface(
        fn=chat_fn,
        type="messages",
        title="Phi-4 Local Analyst",
        description="Prompt-only agent: schema discovery + SQL joins through tools; sandboxed execution.",
        submit_btn="Ask",
        stop_btn="Stop",
        additional_inputs=[files_input],
    )

if __name__ == "__main__":
    ui.launch(server_name="0.0.0.0", server_port=7860, share=False)
