from flask import Flask, jsonify, request, send_from_directory, session, redirect
from flask_cors import CORS
import psycopg2
import psycopg2.extras
import os
import csv
import io
import json
import re
import requests as http_requests
from datetime import datetime, date
from urllib.parse import urlparse

app = Flask(__name__, static_folder='static')
CORS(app)

app.secret_key = os.environ.get('SECRET_KEY', 'jkpsc-cce-2026-change-this')
LOGIN_USERNAME = os.environ.get('LOGIN_USER', 'Samin')
LOGIN_PASSWORD = os.environ.get('LOGIN_PASS', 'NewPassword123')

DATABASE_URL = os.environ.get('DATABASE_URL', '')

GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
GROQ_API_URL = 'https://api.groq.com/openai/v1/chat/completions'
GROQ_MODEL = os.environ.get('GROQ_MODEL', 'llama-3.1-8b-instant')

ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')
ANTHROPIC_API_URL = 'https://api.anthropic.com/v1/messages'
ANTHROPIC_MODEL = 'claude-3-haiku-20240307'

# ── JKPSC subject seed data (replaces RBI subjects) ──
JKPSC_SUBJECTS = [
    (1, 'GS — History, Geography & Culture', '🏛️'),
    (2, 'GS — Polity & Governance', '⚖️'),
    (3, 'GS — Economy, Sci & Environment', '📊'),
    (4, 'J&K GK & Current Affairs', '🏔️'),
    (5, 'CSAT — Aptitude & Reasoning', '🧮'),
    (6, 'Mains — GS & Ethics', '📖'),
]


def get_db():
    try:
        conn = psycopg2.connect(DATABASE_URL, sslmode='require')
        return conn
    except Exception as e:
        app.logger.error(f"DB connection failed: {e}")
        raise


def logged_in():
    return session.get('auth') is True


def init_db():
    try:
        db = get_db()
        cur = db.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS subjects (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                icon VARCHAR(10) DEFAULT '📘'
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS questions (
                id SERIAL PRIMARY KEY,
                subject_id INT NOT NULL,
                question_text TEXT NOT NULL,
                option_a TEXT NOT NULL,
                option_b TEXT NOT NULL,
                option_c TEXT NOT NULL,
                option_d TEXT NOT NULL,
                correct_option CHAR(1) NOT NULL,
                explanation TEXT,
                difficulty VARCHAR(10) DEFAULT 'medium',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (subject_id) REFERENCES subjects(id) ON DELETE CASCADE
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS quiz_attempts (
                id SERIAL PRIMARY KEY,
                subject_id INT,
                total_questions INT DEFAULT 0,
                correct_answers INT DEFAULT 0,
                wrong_answers INT DEFAULT 0,
                skipped INT DEFAULT 0,
                score_percent DECIMAL(5,2) DEFAULT 0,
                time_taken_seconds INT DEFAULT 0,
                attempted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (subject_id) REFERENCES subjects(id) ON DELETE SET NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS attempt_details (
                id SERIAL PRIMARY KEY,
                attempt_id INT NOT NULL,
                question_id INT NOT NULL,
                selected_option CHAR(1),
                is_correct SMALLINT DEFAULT 0,
                FOREIGN KEY (attempt_id) REFERENCES quiz_attempts(id) ON DELETE CASCADE
            )
        """)
        # Seed JKPSC subjects
        cur.execute("SELECT COUNT(*) FROM subjects")
        if cur.fetchone()[0] == 0:
            cur.executemany(
                "INSERT INTO subjects (id, name, icon) VALUES (%s, %s, %s)",
                JKPSC_SUBJECTS
            )
        db.commit()
        cur.close()
        db.close()
        app.logger.info("DB initialised for JKPSC CCE")
    except Exception as e:
        app.logger.warning(f"DB init skipped: {e}")


def call_groq(system_prompt, user_prompt, max_tokens=2048):
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set")
    headers = {
        'Authorization': f'Bearer {GROQ_API_KEY}',
        'Content-Type': 'application/json',
    }
    payload = {
        'model': GROQ_MODEL,
        'max_tokens': max_tokens,
        'temperature': 0.7,
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ],
    }
    try:
        resp = http_requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=25)
        resp.raise_for_status()
        return resp.json()['choices'][0]['message']['content'].strip()
    except http_requests.exceptions.Timeout:
        raise ValueError("Groq API timed out. Try fewer questions.")
    except http_requests.exceptions.RequestException as e:
        raise ValueError(f"Groq API error: {str(e)}")


def call_claude(system_prompt, messages, max_tokens=1024):
    if not ANTHROPIC_API_KEY:
        user_text = messages[-1]['content'] if messages else ''
        return call_groq(system_prompt, user_text, max_tokens=max_tokens)
    headers = {
        'x-api-key': ANTHROPIC_API_KEY,
        'anthropic-version': '2023-06-01',
        'Content-Type': 'application/json',
    }
    payload = {
        'model': ANTHROPIC_MODEL,
        'max_tokens': max_tokens,
        'system': system_prompt,
        'messages': messages,
    }
    resp = http_requests.post(ANTHROPIC_API_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()['content'][0]['text'].strip()


def parse_json_from_ai(text):
    text = re.sub(r'^```(?:json)?\s*', '', text.strip(), flags=re.MULTILINE)
    text = re.sub(r'\s*```$', '', text.strip(), flags=re.MULTILINE)
    start = min(
        (text.find(c) for c in ['[', '{'] if text.find(c) != -1),
        default=0
    )
    return json.loads(text[start:])


def predict_difficulty(question_text, explanation=''):
    combined = (question_text + ' ' + explanation).lower()
    word_count = len(question_text.split())
    hard_signals = ['constitutional', 'amendment', 'article', 'schedule', 'judiciary',
                    'governance', 'international', 'monetary', 'fiscal', 'biodiversity',
                    'dogra', 'accession', 'reorganisation', 'ecology', 'syllogism']
    easy_signals = ['capital of', 'who is', 'when was', 'which of the following',
                    'state symbol', 'national park', 'river', 'valley', 'synonym']
    hard_hits = sum(1 for kw in hard_signals if kw in combined)
    easy_hits = sum(1 for kw in easy_signals if kw in combined)
    if hard_hits >= 2 or word_count > 60:
        return 'hard'
    if easy_hits >= 2 or word_count < 20:
        return 'easy'
    return 'medium'


def dict_fetchall(cur):
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def dict_fetchone(cur):
    cols = [d[0] for d in cur.description]
    row = cur.fetchone()
    return dict(zip(cols, row)) if row else None


# ── HEALTH ──────────────────────────────────────────────────────────────────

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'app': 'JKPSC CCE 2026'}), 200


# ── LOGIN / LOGOUT ──────────────────────────────────────────────────────────

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        u = request.form.get('username', '').strip()
        p = request.form.get('password', '')
        if u == LOGIN_USERNAME and p == LOGIN_PASSWORD:
            session.permanent = True
            session['auth'] = True
            session['username'] = u
            return redirect('/')
        return send_from_directory('static', 'login.html'), 401
    return send_from_directory('static', 'login.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')


# ── AUTH GUARD ──────────────────────────────────────────────────────────────

@app.before_request
def require_login():
    if request.endpoint in ('login', 'logout', 'static', 'health'):
        return
    if not logged_in():
        if request.path.startswith('/api/'):
            return jsonify({'error': 'Unauthorized'}), 401
        return redirect('/login')


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


# ── SUBJECTS ────────────────────────────────────────────────────────────────

@app.route('/api/subjects')
def get_subjects():
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT * FROM subjects ORDER BY id")
    subs = dict_fetchall(cur)
    for s in subs:
        cur.execute("SELECT COUNT(*) FROM questions WHERE subject_id=%s", (s['id'],))
        s['question_count'] = cur.fetchone()[0]
    cur.close(); db.close()
    return jsonify(subs)


# ── QUIZ ─────────────────────────────────────────────────────────────────────

@app.route('/api/quiz/<int:subject_id>')
def get_quiz(subject_id):
    limit = min(int(request.args.get('limit', 10)), 50)
    diff = request.args.get('difficulty', None)
    db = get_db()
    cur = db.cursor()
    if diff:
        cur.execute(
            "SELECT * FROM questions WHERE subject_id=%s AND difficulty=%s ORDER BY RANDOM() LIMIT %s",
            (subject_id, diff, limit)
        )
    else:
        cur.execute(
            "SELECT * FROM questions WHERE subject_id=%s ORDER BY RANDOM() LIMIT %s",
            (subject_id, limit)
        )
    qs = dict_fetchall(cur)
    for q in qs:
        q.pop('correct_option', None)
        q.pop('explanation', None)
    cur.close(); db.close()
    return jsonify(qs)


@app.route('/api/quiz/submit', methods=['POST'])
def submit_quiz():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Invalid JSON'}), 400
    subject_id = data.get('subject_id')
    answers = data.get('answers', {})
    time_taken = int(data.get('time_taken_seconds', 0))
    # Support inline questions (AI-generated, not stored in DB)
    inline_questions = {str(q['id']): q for q in data.get('questions', [])}

    db = get_db()
    cur = db.cursor()
    correct = wrong = skipped = 0
    details = []

    for qid_str, selected in answers.items():
        correct_opt = None
        explanation = ''

        if qid_str in inline_questions:
            # AI-generated question — answer info comes from client
            q = inline_questions[qid_str]
            correct_opt = q.get('correct_option', '').upper()
            explanation = q.get('explanation', '')
            qid = None
        else:
            try:
                qid = int(qid_str)
            except ValueError:
                continue
            cur.execute("SELECT correct_option, explanation FROM questions WHERE id=%s", (qid,))
            row = cur.fetchone()
            if not row:
                continue
            correct_opt, explanation = row

        if not selected:
            skipped += 1
            details.append({'qid': qid_str, 'selected': None, 'is_correct': 0,
                            'correct': correct_opt, 'explanation': explanation or ''})
        elif selected.upper() == correct_opt.upper():
            correct += 1
            details.append({'qid': qid_str, 'selected': selected, 'is_correct': 1,
                            'correct': correct_opt, 'explanation': explanation or ''})
        else:
            wrong += 1
            details.append({'qid': qid_str, 'selected': selected, 'is_correct': 0,
                            'correct': correct_opt, 'explanation': explanation or ''})

    total = correct + wrong + skipped
    score_pct = round(correct / total * 100, 2) if total else 0

    cur.execute(
        """INSERT INTO quiz_attempts
           (subject_id, total_questions, correct_answers, wrong_answers, skipped, score_percent, time_taken_seconds)
           VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
        (subject_id, total, correct, wrong, skipped, score_pct, time_taken)
    )
    attempt_id = cur.fetchone()[0]

    # Only store attempt_details for DB-backed questions (integer IDs)
    for d in details:
        try:
            db_qid = int(d['qid'])
            cur.execute(
                "INSERT INTO attempt_details (attempt_id, question_id, selected_option, is_correct) VALUES (%s,%s,%s,%s)",
                (attempt_id, db_qid, d['selected'], d['is_correct'])
            )
        except (ValueError, TypeError):
            pass  # skip inline AI questions

    db.commit(); cur.close(); db.close()
    return jsonify({
        'attempt_id': attempt_id, 'total': total, 'correct': correct,
        'wrong': wrong, 'skipped': skipped, 'score_percent': score_pct, 'details': details
    })


# ── HISTORY ─────────────────────────────────────────────────────────────────

@app.route('/api/history')
def get_history():
    limit = min(int(request.args.get('limit', 20)), 100)
    db = get_db()
    cur = db.cursor()
    cur.execute("""
        SELECT qa.*, s.name as subject_name, s.icon
        FROM quiz_attempts qa
        LEFT JOIN subjects s ON qa.subject_id = s.id
        ORDER BY qa.attempted_at DESC LIMIT %s
    """, (limit,))
    rows = dict_fetchall(cur)
    for r in rows:
        if r.get('attempted_at'):
            r['attempted_at'] = str(r['attempted_at'])
        if r.get('score_percent') is not None:
            r['score_percent'] = float(r['score_percent'])
    cur.close(); db.close()
    return jsonify(rows)


# ── STATS ────────────────────────────────────────────────────────────────────

@app.route('/api/stats')
def get_stats():
    db = get_db()
    cur = db.cursor()
    cur.execute(
        "SELECT COUNT(*) as total_attempts, AVG(score_percent) as avg_score, MAX(score_percent) as best_score FROM quiz_attempts"
    )
    stats = dict_fetchone(cur)
    cur.execute("SELECT COUNT(*) as total_questions FROM questions")
    stats['total_questions'] = cur.fetchone()[0]
    cur.close(); db.close()
    stats['avg_score'] = round(float(stats['avg_score']), 1) if stats['avg_score'] else 0
    stats['best_score'] = round(float(stats['best_score']), 1) if stats['best_score'] else 0
    return jsonify(stats)


# ── ANALYTICS ────────────────────────────────────────────────────────────────

@app.route('/api/analytics')
def performance_analytics():
    db = get_db()
    cur = db.cursor()
    cur.execute("""
        SELECT s.id, s.name, s.icon,
               COUNT(qa.id) AS attempts,
               AVG(qa.score_percent) AS avg_score,
               MAX(qa.score_percent) AS best_score,
               SUM(qa.correct_answers) AS total_correct,
               SUM(qa.total_questions) AS total_answered
        FROM subjects s
        LEFT JOIN quiz_attempts qa ON s.id = qa.subject_id
        GROUP BY s.id, s.name, s.icon
        ORDER BY avg_score DESC NULLS LAST
    """)
    subject_stats = dict_fetchall(cur)
    for s in subject_stats:
        s['avg_score'] = round(float(s['avg_score'] or 0), 1)
        s['best_score'] = round(float(s['best_score'] or 0), 1)
        s['accuracy'] = round(float(s['total_correct'] or 0) / float(s['total_answered'] or 1) * 100, 1)

    cur.execute("""
        SELECT qa.attempted_at, qa.score_percent, s.name as subject_name
        FROM quiz_attempts qa
        LEFT JOIN subjects s ON qa.subject_id = s.id
        ORDER BY qa.attempted_at DESC LIMIT 20
    """)
    trend = dict_fetchall(cur)
    for t in trend:
        t['score_percent'] = float(t['score_percent'] or 0)
        if t.get('attempted_at'):
            t['attempted_at'] = str(t['attempted_at'])
    trend.reverse()

    cur.execute(
        "SELECT COUNT(*) as total_attempts, AVG(score_percent) as overall_avg, "
        "SUM(correct_answers) as total_correct, SUM(wrong_answers) as total_wrong, "
        "SUM(skipped) as total_skipped FROM quiz_attempts"
    )
    overall = dict_fetchone(cur)
    overall['overall_avg'] = round(float(overall['overall_avg'] or 0), 1)

    cur.close(); db.close()
    return jsonify({
        'overall': overall,
        'subject_stats': subject_stats,
        'score_trend': trend,
        'weak_subjects': [s['name'] for s in subject_stats if s['accuracy'] < 50],
        'strong_subjects': [s['name'] for s in subject_stats if s['accuracy'] >= 75],
    })


# ── QUESTIONS CRUD ───────────────────────────────────────────────────────────

@app.route('/api/questions', methods=['POST'])
def add_question():
    d = request.get_json(silent=True)
    if not d:
        return jsonify({'error': 'Invalid JSON'}), 400
    required = ['subject_id', 'question_text', 'option_a', 'option_b', 'option_c', 'option_d', 'correct_option']
    for field in required:
        if not d.get(field):
            return jsonify({'error': f'Missing field: {field}'}), 400
    if d['correct_option'].upper() not in ('A', 'B', 'C', 'D'):
        return jsonify({'error': 'correct_option must be A, B, C or D'}), 400
    db = get_db()
    cur = db.cursor()
    cur.execute(
        "INSERT INTO questions (subject_id,question_text,option_a,option_b,option_c,option_d,"
        "correct_option,explanation,difficulty) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
        (int(d['subject_id']), d['question_text'].strip(), d['option_a'].strip(),
         d['option_b'].strip(), d['option_c'].strip(), d['option_d'].strip(),
         d['correct_option'].upper(), d.get('explanation', '').strip(),
         d.get('difficulty', 'medium'))
    )
    new_id = cur.fetchone()[0]
    db.commit(); cur.close(); db.close()
    return jsonify({'success': True, 'id': new_id}), 201


@app.route('/api/questions')
def get_questions():
    subject_id = request.args.get('subject_id')
    limit = min(int(request.args.get('limit', 50)), 200)
    db = get_db()
    cur = db.cursor()
    if subject_id:
        cur.execute("SELECT * FROM questions WHERE subject_id=%s ORDER BY id DESC LIMIT %s", (subject_id, limit))
    else:
        cur.execute("SELECT * FROM questions ORDER BY id DESC LIMIT %s", (limit,))
    qs = dict_fetchall(cur)
    cur.close(); db.close()
    return jsonify(qs)


@app.route('/api/questions/<int:qid>', methods=['DELETE'])
def delete_question(qid):
    db = get_db()
    cur = db.cursor()
    cur.execute("DELETE FROM questions WHERE id=%s", (qid,))
    db.commit(); cur.close(); db.close()
    return jsonify({'success': True})


# ── AI: GENERATE QUESTIONS (Groq — JKPSC context) ────────────────────────────

@app.route('/api/ai/generate', methods=['POST'])
def ai_generate_questions():
    data = request.get_json(silent=True) or {}
    topic = data.get('topic', '').strip()
    num_questions = min(int(data.get('num_questions', 5)), 10)
    subject_id = data.get('subject_id')
    save_to_db = bool(data.get('save', False))

    if not topic:
        return jsonify({'error': 'topic is required'}), 400

    system_prompt = (
        "You are an expert JKPSC CCE 2026 exam question creator for Jammu & Kashmir. "
        "Always respond with a valid JSON array only, no preamble, no markdown."
    )
    user_prompt = (
        f"Generate {num_questions} JKPSC CCE Prelims multiple-choice questions on: \"{topic}\".\n\n"
        "Return a JSON array where every element has exactly these keys:\n"
        "question_text, option_a, option_b, option_c, option_d, "
        "correct_option (A/B/C/D), explanation, difficulty (easy/medium/hard)\n\n"
        "Focus on J&K History, Geography, Polity, Indian Constitution, J&K Governance, "
        "Economy, Current Affairs as relevant. Questions must be factually accurate and "
        "exam-appropriate for JKPSC CCE Prelims level."
    )

    try:
        raw = call_groq(system_prompt, user_prompt, max_tokens=1500)
        questions = parse_json_from_ai(raw)
    except ValueError as e:
        return jsonify({'error': str(e)}), 503
    except Exception as e:
        return jsonify({'error': 'AI service error', 'detail': str(e)}), 502

    valid = []
    for q in questions:
        if not all(k in q for k in ['question_text', 'option_a', 'option_b', 'option_c', 'option_d', 'correct_option']):
            continue
        q['correct_option'] = q['correct_option'].upper().strip('.')
        if q['correct_option'] not in ('A', 'B', 'C', 'D'):
            continue
        if not q.get('explanation'):
            q['explanation'] = ''
        if q.get('difficulty') not in ('easy', 'medium', 'hard'):
            q['difficulty'] = predict_difficulty(q['question_text'], q.get('explanation', ''))
        valid.append(q)

    saved_ids = []
    if save_to_db and subject_id and valid:
        db = get_db()
        cur = db.cursor()
        for q in valid:
            cur.execute(
                "INSERT INTO questions (subject_id,question_text,option_a,option_b,option_c,"
                "option_d,correct_option,explanation,difficulty) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
                (int(subject_id), q['question_text'], q['option_a'], q['option_b'],
                 q['option_c'], q['option_d'], q['correct_option'],
                 q.get('explanation', ''), q['difficulty'])
            )
            saved_ids.append(cur.fetchone()[0])
        db.commit(); cur.close(); db.close()

    return jsonify({
        'generated': len(valid), 'saved': len(saved_ids),
        'saved_ids': saved_ids, 'questions': valid
    })


# ── AI: TUTOR CHAT (Claude → Groq fallback, JKPSC context) ───────────────────

@app.route('/api/ai/tutor', methods=['POST'])
def ai_tutor():
    data = request.get_json(silent=True) or {}
    message = data.get('message', '').strip()
    history = data.get('history', [])
    if not message:
        return jsonify({'error': 'message is required'}), 400

    system_prompt = (
        "You are an expert JKPSC CCE 2026 tutor helping a student from Kashmir prepare "
        "for the KAS exam. You specialise in: J&K History (Dogra period, Accession 1947, "
        "Article 370 abrogation), Indian Polity (M. Laxmikanth), Geography (J&K valleys, "
        "rivers, passes), Indian Economy, Environment & Ecology (National Parks of J&K), "
        "General Awareness, and CSAT. Give step-by-step answers, be encouraging, "
        "include J&K-specific examples where relevant, and keep answers exam-focused."
    )

    messages = []
    for h in history[-10:]:
        role = h.get('role', '')
        content = h.get('content', '')
        if role in ('user', 'assistant') and content:
            messages.append({'role': role, 'content': content})
    messages.append({'role': 'user', 'content': message})

    try:
        reply = call_claude(system_prompt, messages, max_tokens=1024)
    except Exception as e:
        return jsonify({'error': 'AI service error', 'detail': str(e)}), 502

    return jsonify({'reply': reply})


# ── AI: EXPLAIN ───────────────────────────────────────────────────────────────

@app.route('/api/ai/explain', methods=['POST'])
def ai_generate_explanation():
    data = request.get_json(silent=True) or {}
    q = data  # accept full question object or question_id
    qid = data.get('question_id')
    db = None

    if qid:
        db = get_db()
        cur = db.cursor()
        cur.execute("SELECT * FROM questions WHERE id=%s", (int(qid),))
        q = dict_fetchone(cur)
        if not q:
            cur.close(); db.close()
            return jsonify({'error': 'Question not found'}), 404
        if q.get('explanation', '').strip():
            cur.close(); db.close()
            return jsonify({'explanation': q['explanation'], 'cached': True})

    required = ['question_text', 'option_a', 'option_b', 'option_c', 'option_d', 'correct_option']
    if not all(q.get(k) for k in required):
        return jsonify({'error': 'Provide question_id OR full question fields'}), 400

    prompt = (
        f"Question: {q['question_text']}\n"
        f"A) {q['option_a']}  B) {q['option_b']}  C) {q['option_c']}  D) {q['option_d']}\n"
        f"Correct Answer: {q['correct_option']}\n\n"
        "Write a clear, concise explanation (2-4 sentences) relevant to JKPSC CCE Prelims, "
        "including any J&K-specific context where applicable."
    )

    try:
        explanation = call_groq("You are an expert JKPSC CCE tutor.", prompt, max_tokens=400)
    except Exception as e:
        return jsonify({'error': 'AI service error', 'detail': str(e)}), 502

    if qid and db:
        cur.execute("UPDATE questions SET explanation=%s WHERE id=%s", (explanation, int(qid)))
        db.commit(); cur.close(); db.close()

    return jsonify({'explanation': explanation, 'cached': False})


# ── AI: HINT ──────────────────────────────────────────────────────────────────

@app.route('/api/ai/hint', methods=['POST'])
def ai_get_hint():
    data = request.get_json(silent=True) or {}
    qid = data.get('question_id')
    if qid:
        db = get_db()
        cur = db.cursor()
        cur.execute("SELECT * FROM questions WHERE id=%s", (int(qid),))
        q = dict_fetchone(cur)
        cur.close(); db.close()
        if not q:
            return jsonify({'error': 'Question not found'}), 404
    else:
        q = data
    if not q.get('question_text'):
        return jsonify({'error': 'question_id or question_text required'}), 400

    prompt = (
        f"Question: {q['question_text']}\n"
        f"A) {q.get('option_a','')}  B) {q.get('option_b','')}  "
        f"C) {q.get('option_c','')}  D) {q.get('option_d','')}\n\n"
        "Give ONE helpful hint without revealing the answer. 1-2 sentences. "
        "Relate to JKPSC CCE context if possible."
    )
    try:
        hint = call_groq("You are a JKPSC CCE tutor giving helpful hints.", prompt, max_tokens=200)
    except Exception as e:
        return jsonify({'error': 'AI service error', 'detail': str(e)}), 502
    return jsonify({'hint': hint})


# ── AI: STUDY PLAN ────────────────────────────────────────────────────────────

@app.route('/api/ai/study-plan', methods=['POST'])
def ai_study_plan():
    data = request.get_json(silent=True) or {}
    exam_date = data.get('exam_date', 'November/December 2026')
    hours_per_day = int(data.get('hours_per_day', 4))
    weak_subjects = data.get('weak_subjects', [])
    weak_str = ', '.join(weak_subjects) if weak_subjects else 'None identified yet'

    user_prompt = (
        f"Create a detailed 9-month JKPSC CCE 2026 study plan. "
        f"Exam: {exam_date}. Hours/day: {hours_per_day}. Weak subjects: {weak_str}. "
        "Include phase-wise breakdown (Foundation → Core → Current Affairs → Mock Blitz → Revision), "
        "daily schedule, recommended books, and J&K-specific preparation tips."
    )
    try:
        plan = call_claude(
            "You are an expert JKPSC CCE 2026 exam coach specialising in J&K History, "
            "Indian Polity, Geography, Economy, and KAS exam strategy.",
            [{'role': 'user', 'content': user_prompt}],
            max_tokens=1500
        )
    except Exception as e:
        return jsonify({'error': 'AI service error', 'detail': str(e)}), 502

    return jsonify({'plan': plan, 'weak_subjects': weak_subjects})


# ── AI: STATUS ────────────────────────────────────────────────────────────────

@app.route('/api/ai/status')
def ai_status():
    groq_ok = bool(GROQ_API_KEY)
    anthropic_ok = bool(ANTHROPIC_API_KEY)
    tutor_provider = 'claude' if anthropic_ok else ('groq' if groq_ok else 'none')
    return jsonify({
        'groq': {'configured': groq_ok, 'model': GROQ_MODEL},
        'claude': {'configured': anthropic_ok},
        'tutor_provider': tutor_provider,
        'all_features_active': groq_ok,
        'app': 'JKPSC CCE 2026'
    })


@app.route('/api/ai/test')
def ai_test():
    try:
        result = call_groq("You are a helpful assistant.", "Say hello in one word.", max_tokens=10)
        return jsonify({'success': True, 'response': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ── CSV IMPORT ────────────────────────────────────────────────────────────────

@app.route('/api/questions/import', methods=['POST'])
def import_questions_csv():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    subject_id = request.form.get('subject_id')
    if not subject_id:
        return jsonify({'error': 'subject_id is required'}), 400
    f = request.files['file']
    stream = io.StringIO(f.stream.read().decode('utf-8-sig'), newline=None)
    reader = csv.DictReader(stream)
    required_cols = {'question_text', 'option_a', 'option_b', 'option_c', 'option_d', 'correct_option'}
    if not required_cols.issubset(set(reader.fieldnames or [])):
        missing = required_cols - set(reader.fieldnames or [])
        return jsonify({'error': f'Missing CSV columns: {missing}'}), 400
    db = get_db()
    cur = db.cursor()
    inserted = skipped = 0
    for row in reader:
        co = row.get('correct_option', '').strip().upper()
        if co not in ('A', 'B', 'C', 'D'):
            skipped += 1; continue
        diff = row.get('difficulty', '').strip().lower()
        if diff not in ('easy', 'medium', 'hard'):
            diff = predict_difficulty(row['question_text'], row.get('explanation', ''))
        try:
            cur.execute(
                "INSERT INTO questions (subject_id,question_text,option_a,option_b,option_c,"
                "option_d,correct_option,explanation,difficulty) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (int(subject_id), row['question_text'].strip(), row['option_a'].strip(),
                 row['option_b'].strip(), row['option_c'].strip(), row['option_d'].strip(),
                 co, row.get('explanation', '').strip(), diff)
            )
            inserted += 1
        except Exception:
            skipped += 1
    db.commit(); cur.close(); db.close()
    return jsonify({'inserted': inserted, 'skipped': skipped}), 201


# ── ERROR HANDLERS ────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Not found'}), 404
    return send_from_directory('static', 'index.html')


@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500


# ── STARTUP ───────────────────────────────────────────────────────────────────

try:
    init_db()
except Exception as e:
    app.logger.warning(f"DB init failed: {e}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
