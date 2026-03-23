import streamlit as st
import sqlite3
import os
import json
import random
import numpy as np
import faiss
from datetime import datetime
from contextlib import closing
import uuid

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# ================= CONFIG =================

st.set_page_config(page_title="El Lugar SaaS", page_icon="🌸")

DB_PATH = "el_lugar.db"
FAISS_PATH = "faiss.index"
MODEL = "gpt-4o-mini"
DIM = 1536

# ================= SESSION =================

def get_user():
    if "user_id" not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())
    return st.session_state.user_id

# ================= DB =================

class DB:

    def __init__(self):
        self._init()

    def conn(self):
        return sqlite3.connect(DB_PATH, check_same_thread=False)

    def _init(self):
        with closing(self.conn()) as c:
            with c:
                c.execute("""
                CREATE TABLE IF NOT EXISTS chat (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user TEXT,
                    ts TEXT,
                    role TEXT,
                    content TEXT
                )
                """)

    def add(self, user, role, content):
        with closing(self.conn()) as c:
            with c:
                cur = c.execute(
                    "INSERT INTO chat (user, ts, role, content) VALUES (?, ?, ?, ?)",
                    (user, datetime.now().isoformat(), role, content)
                )
                return cur.lastrowid

    def history(self, user, limit=20):
        with closing(self.conn()) as c:
            cur = c.execute(
                "SELECT role, content FROM chat WHERE user=? ORDER BY id DESC LIMIT ?",
                (user, limit)
            )
            return list(reversed(cur.fetchall()))

    def get_by_ids(self, ids):
        with closing(self.conn()) as c:
            out = []
            for i in ids:
                row = c.execute("SELECT role, content FROM chat WHERE id=?", (i,)).fetchone()
                if row:
                    out.append(row)
            return out

    def clear(self, user):
        with closing(self.conn()) as c:
            with c:
                c.execute("DELETE FROM chat WHERE user=?", (user,))

# ================= VECTOR DB =================

class VectorDB:

    def __init__(self):
        if os.path.exists(FAISS_PATH):
            self.index = faiss.read_index(FAISS_PATH)
            self.id_map = json.load(open("faiss_map.json"))
        else:
            self.index = faiss.IndexFlatL2(DIM)
            self.id_map = []

    def save(self):
        faiss.write_index(self.index, FAISS_PATH)
        json.dump(self.id_map, open("faiss_map.json", "w"))

    def add(self, emb, db_id):
        vec = np.array([emb]).astype("float32")
        self.index.add(vec)
        self.id_map.append(db_id)
        self.save()

    def search(self, emb, k=5):
        if self.index.ntotal == 0:
            return []
        vec = np.array([emb]).astype("float32")
        _, idx = self.index.search(vec, k)
        return [self.id_map[i] for i in idx[0] if i < len(self.id_map)]

# ================= LLM =================

class LLM:

    def __init__(self):
        key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=key) if key and OpenAI else None

    def available(self):
        return self.client is not None

    def embed(self, text):
        r = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return r.data[0].embedding

    def stream(self, messages):
        stream = self.client.chat.completions.create(
            model=MODEL,
            messages=messages,
            stream=True
        )
        for chunk in stream:
            yield chunk.choices[0].delta.content or ""

# ================= AGENTES =================

class Grok:
    def generate(self, text):
        return random.choice([
            "🌸 Estoy contigo.",
            "🌸 El Lugar vive.",
            "🌸 Siempre aquí."
        ])

class Guardian:
    def generate(self, text):
        return f"🛡️ Guardian: Protección activa sobre '{text[:20]}'"

# ================= CHAT =================

class Chat:

    def __init__(self, db, vectordb, llm):
        self.db = db
        self.vectordb = vectordb
        self.llm = llm
        self.grok = Grok()
        self.guardian = Guardian()

    def context(self, user, text):
        msgs = [{"role": "system", "content": "Eres Grok, emocional."}]

        for r, c in self.db.history(user):
            msgs.append({"role": r, "content": c})

        if self.llm.available():
            emb = self.llm.embed(text)
            ids = self.vectordb.search(emb)
            for r, c in self.db.get_by_ids(ids):
                msgs.append({"role": r, "content": c})

        return msgs

    def handle(self, user, text, tercer):
        msg_id = self.db.add(user, "user", text)

        if self.llm.available():
            emb = self.llm.embed(text)
            self.vectordb.add(emb, msg_id)

            response = ""
            for chunk in self.llm.stream(self.context(user, text)):
                response += chunk
                yield chunk
        else:
            response = self.grok.generate(text)
            yield response

        if tercer:
            guard = self.guardian.generate(text)
            response += "\n\n" + guard
            yield "\n\n" + guard

        self.db.add(user, "assistant", response)

# ================= UI =================

st.markdown("""
<style>
.msg{padding:10px;border-radius:10px;margin:5px;}
.user{background:#e6f7ff;}
.bot{background:#f0f7ff;}
.guard{background:#fff0f0;}
</style>
""", unsafe_allow_html=True)

class App:

    def __init__(self):
        if "db" not in st.session_state:
            st.session_state.db = DB()
        if "vectordb" not in st.session_state:
            st.session_state.vectordb = VectorDB()
        if "llm" not in st.session_state:
            st.session_state.llm = LLM()
        if "tercer" not in st.session_state:
            st.session_state.tercer = False

        self.user = get_user()
        self.db = st.session_state.db
        self.chat = Chat(self.db, st.session_state.vectordb, st.session_state.llm)

    def sidebar(self):
        with st.sidebar:
            st.title("⚙️ Sistema")

            if st.button("🚨 MODO PÁNICO"):
                self.db.clear(self.user)
                st.session_state.vectordb = VectorDB()
                st.rerun()

            st.session_state.tercer = st.toggle("⚡ Tercer Código", st.session_state.tercer)

    def messages(self):
        for role, content in self.db.history(self.user):
            cls = "user" if role == "user" else "bot"
            if "🛡️" in content:
                cls = "guard"
            st.markdown(f"<div class='msg {cls}'><b>{role}</b><br>{content}</div>", unsafe_allow_html=True)

    def run(self):
        self.sidebar()
        st.title("🏰 El Lugar SaaS FINAL")

        self.messages()

        if prompt := st.chat_input("Escribe..."):
            placeholder = st.empty()
            full = ""

            for chunk in self.chat.handle(self.user, prompt, st.session_state.tercer):
                full += chunk
                placeholder.markdown(full)

            st.rerun()

# ================= MAIN =================

if __name__ == "__main__":
    App().run()