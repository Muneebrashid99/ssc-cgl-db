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

app.secret_key = os.environ.get('SECRET_KEY', 'ssc-cgl-secret-2026-change-this')
LOGIN_USERNAME = os.environ.get('LOGIN_USER', 'Samin')
LOGIN_PASSWORD = os.environ.get('LOGIN_PASS', 'NewPassword123')

DATABASE_URL = os.environ.get('DATABASE_URL', '')

GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
GROQ_API_URL = 'https://api.groq.com/openai/v1/chat/completions'
GROQ_MODEL = 'llama-3.3-70b-versatile'

ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')
ANTHROPIC_API_URL = 'https://api.anthropic.com/v1/messages'
ANTHROPIC_MODEL = 'claude-3-haiku-20240307'


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
        cur.execute("SELECT COUNT(*) FROM subjects")
        if cur.fetchone()[0] == 0:
            seeds = [
                (1, 'Quantitative Aptitude', '📐'),
                (2, 'General Intelligence & Reasoning', '🧠'),
                (3, 'English Language', '📝'),
                (4, 'General Awareness', '🌍'),
            ]
            cur.executemany("INSERT INTO subjects (id,name,icon) VALUES (%s,%s,%s)", seeds)
        db.commit()
        cur.close()
        db.close()
        app.logger.info("DB initialised")
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
    resp = http_requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()['choices'][0]['message']['content'].strip()


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
    hard_signals = ['probability','permutation','combination','integration','differentiation','compound interest','mixture','alligation','mensuration','trigonometry','logarithm','series','sequence','inference','syllogism','analytical','crypt']
    easy_signals = ['simple','basic','find the value','which of the following','synonym','antonym','fill in the blank','capital of','who is','when was','one word']
    hard_hits = sum(1 for kw in hard_signals if kw in combined)
    easy_hits = sum(1 for kw in easy_signals if kw in combined)
    if hard_hits >= 2 or word_count > 60:
        return 'hard'
    if easy_hits >= 2 or word_count < 20:
        return 'easy'
    return 'medium'


def get_recently_seen_ids(subject_id, lookback=5):
    try:
        db = get_db()
        cur = db.cursor()
        cur.execute("SELECT id FROM quiz_attempts WHERE subject_id = %s ORDER BY attempted_at DESC LIMIT %s", (subject_id, lookback))
        attempt_ids = [r[0] for r in cur.fetchall()]
        if not attempt_ids:
            cur.close(); db.close()
            return []
        cur.execute("SELECT DISTINCT question_id FROM attempt_details WHERE attempt_id = ANY(%s)", (attempt_ids,))
        seen = [r[0] for r in cur.fetchall()]
        cur.close(); db.close()
        return seen
    except Exception:
        return []


def dict_fetchall(cur):
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def dict_fetchone(cur):
    cols = [d[0] for d in cur.description]
    row = cur.fetchone()
    return dict(zip(cols, row)) if row else None


@app.route('/health')
def health():
    return jsonify({'status': 'ok'}), 200


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = ''
    if request.method == 'POST':
        u = request.form.get('username', '').strip()
        p = request.form.get('password', '')
        if u == LOGIN_USERNAME and p == LOGIN_PASSWORD:
            session.permanent = True
            session['auth'] = True
            return redirect('/')
        error = 'Invalid credentials. Try again.'
    err_html = f"<div class='err'>&#9888;&#65039; {error}</div>" if error else ""
    page = """<!DOCTYPE html>
<html>
<head>
<title>SSC CGL Login</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;600&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0;}
body{background:#05070f;font-family:'DM Sans',sans-serif;display:flex;align-items:center;justify-content:center;min-height:100vh;}
.box{background:#0f1324;border:1px solid #1c2035;border-radius:20px;padding:44px;width:380px;text-align:center;box-shadow:0 20px 60px rgba(0,0,0,0.5);}
.logo{font-size:3rem;margin-bottom:10px;}
h2{color:#eef0f8;font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;margin-bottom:4px;}
p{color:#4a5270;font-size:12px;margin-bottom:32px;letter-spacing:0.5px;}
label{display:block;text-align:left;font-size:10px;color:#4a5270;font-weight:700;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:6px;}
input{width:100%;background:#090c18;border:1px solid #1c2035;border-radius:10px;color:#eef0f8;font-family:'DM Sans',sans-serif;font-size:14px;padding:13px 16px;outline:none;margin-bottom:16px;transition:border-color 0.2s;}
input:focus{border-color:rgba(245,200,66,0.6);}
button{width:100%;background:linear-gradient(135deg,#f5c842,#e8a020);color:#000;border:none;border-radius:10px;padding:14px;font-family:'Syne',sans-serif;font-size:14px;font-weight:800;cursor:pointer;margin-top:4px;letter-spacing:0.5px;transition:opacity 0.2s;}
button:hover{opacity:0.88;}
.err{color:#ff5757;font-size:12px;margin-bottom:16px;background:rgba(255,87,87,0.08);padding:10px 14px;border-radius:8px;border:1px solid rgba(255,87,87,0.2);}
.foot{margin-top:22px;font-size:11px;color:#2e3450;}
</style>
</head>
<body>
<div class="box">
  <div class="logo">&#9889;</div>
  <h2>SSC CGL 2026</h2>
  <p>Master Prep Hub &mdash; Private Access</p>
  """ + err_html + """
  <form method="POST">
    <label>Username</label>
    <input type="text" name="username" placeholder="Enter username" required autofocus autocomplete="username">
    <label>Password</label>
    <input type="password" name="password" placeholder="Enter password" required autocomplete="current-password">
    <button type="submit">Login to Dashboard</button>
  </form>
  <div class="foot">Your personal SSC CGL 2026 prep dashboard</div>
</div>
</body>
</html>"""
    return page


@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')


@app.before_request
def require_login():
    if request.endpoint in ('login', 'static', 'health'):
        return
    if not logged_in():
        if request.path.startswith('/api/'):
            return jsonify({'error': 'Unauthorized'}), 401
        return redirect('/login')


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


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


@app.route('/api/quiz/<int:subject_id>')
def get_quiz(subject_id):
    limit = min(int(request.args.get('limit', 10)), 50)
    diff = request.args.get('difficulty', None)
    db = get_db()
    cur = db.cursor()
    if diff:
        cur.execute("SELECT * FROM questions WHERE subject_id=%s AND difficulty=%s ORDER BY RANDOM() LIMIT %s", (subject_id, diff, limit))
    else:
        cur.execute("SELECT * FROM questions WHERE subject_id=%s ORDER BY RANDOM() LIMIT %s", (subject_id, limit))
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
    db = get_db()
    cur = db.cursor()
    correct = wrong = skipped = 0
    details = []
    for qid_str, selected in answers.items():
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
            details.append({'qid': qid, 'selected': None, 'is_correct': 0, 'correct': correct_opt, 'explanation': explanation or ''})
        elif selected.upper() == correct_opt.upper():
            correct += 1
            details.append({'qid': qid, 'selected': selected, 'is_correct': 1, 'correct': correct_opt, 'explanation': explanation or ''})
        else:
            wrong += 1
            details.append({'qid': qid, 'selected': selected, 'is_correct': 0, 'correct': correct_opt, 'explanation': explanation or ''})
    total = correct + wrong + skipped
    score_pct = round(correct / total * 100, 2) if total else 0
    cur.execute("INSERT INTO quiz_attempts (subject_id, total_questions, correct_answers, wrong_answers, skipped, score_percent, time_taken_seconds) VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING id",
        (subject_id, total, correct, wrong, skipped, score_pct, time_taken))
    attempt_id = cur.fetchone()[0]
    for d in details:
        cur.execute("INSERT INTO attempt_details (attempt_id, question_id, selected_option, is_correct) VALUES (%s,%s,%s,%s)",
            (attempt_id, d['qid'], d['selected'], d['is_correct']))
    db.commit(); cur.close(); db.close()
    return jsonify({'attempt_id': attempt_id, 'total': total, 'correct': correct, 'wrong': wrong, 'skipped': skipped, 'score_percent': score_pct, 'details': details})


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


@app.route('/api/stats')
def get_stats():
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT COUNT(*) as total_attempts, AVG(score_percent) as avg_score, MAX(score_percent) as best_score FROM quiz_attempts")
    stats = dict_fetchone(cur)
    cur.execute("SELECT COUNT(*) as total_questions FROM questions")
    stats['total_questions'] = cur.fetchone()[0]
    cur.close(); db.close()
    stats['avg_score'] = round(float(stats['avg_score']), 1) if stats['avg_score'] else 0
    stats['best_score'] = round(float(stats['best_score']), 1) if stats['best_score'] else 0
    return jsonify(stats)


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
    cur.execute("INSERT INTO questions (subject_id,question_text,option_a,option_b,option_c,option_d,correct_option,explanation,difficulty) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
        (int(d['subject_id']), d['question_text'].strip(), d['option_a'].strip(), d['option_b'].strip(), d['option_c'].strip(), d['option_d'].strip(), d['correct_option'].upper(), d.get('explanation','').strip(), d.get('difficulty','medium')))
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


@app.route('/api/ai/generate', methods=['POST'])
def ai_generate_questions():
    data = request.get_json(silent=True) or {}
    topic = data.get('topic', '').strip()
    num_questions = min(int(data.get('num_questions', 5)), 20)
    subject_id = data.get('subject_id')
    save_to_db = bool(data.get('save', False))
    if not topic:
        return jsonify({'error': 'topic is required'}), 400
    system_prompt = "You are an expert SSC CGL exam question creator. Always respond with a valid JSON array only, no preamble, no markdown."
    user_prompt = (
        f"Generate {num_questions} SSC CGL multiple-choice questions on the topic: \"{topic}\".\n\n"
        "Return a JSON array where every element has exactly these keys:\n"
        "question_text, option_a, option_b, option_c, option_d, correct_option (A/B/C/D), explanation, difficulty (easy/medium/hard)\n\n"
        "Ensure questions are factually correct, distinct, and exam-appropriate."
    )
    try:
        raw = call_groq(system_prompt, user_prompt, max_tokens=3000)
        questions = parse_json_from_ai(raw)
    except ValueError as e:
        return jsonify({'error': str(e)}), 503
    except Exception as e:
        return jsonify({'error': 'AI service error', 'detail': str(e)}), 502
    valid = []
    for q in questions:
        if not all(k in q for k in ['question_text','option_a','option_b','option_c','option_d','correct_option']):
            continue
        q['correct_option'] = q['correct_option'].upper().strip('.')
        if q['correct_option'] not in ('A','B','C','D'):
            continue
        if not q.get('explanation'):
            q['explanation'] = ''
        if q.get('difficulty') not in ('easy','medium','hard'):
            q['difficulty'] = predict_difficulty(q['question_text'], q.get('explanation',''))
        valid.append(q)
    saved_ids = []
    if save_to_db and subject_id and valid:
        db = get_db()
        cur = db.cursor()
        for q in valid:
            cur.execute("INSERT INTO questions (subject_id,question_text,option_a,option_b,option_c,option_d,correct_option,explanation,difficulty) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
                (int(subject_id), q['question_text'], q['option_a'], q['option_b'], q['option_c'], q['option_d'], q['correct_option'], q.get('explanation',''), q['difficulty']))
            saved_ids.append(cur.fetchone()[0])
        db.commit(); cur.close(); db.close()
    return jsonify({'generated': len(valid), 'saved': len(saved_ids), 'saved_ids': saved_ids, 'questions': valid})


@app.route('/api/quiz/topic')
def quiz_by_topic():
    topic = request.args.get('topic', '').strip()
    subject_id = request.args.get('subject_id')
    difficulty = request.args.get('difficulty')
    limit = min(int(request.args.get('limit', 10)), 50)
    if not topic:
        return jsonify({'error': 'topic param required'}), 400
    db = get_db()
    cur = db.cursor()
    conditions = ["(question_text ILIKE %s OR explanation ILIKE %s)"]
    params = [f'%{topic}%', f'%{topic}%']
    if subject_id:
        conditions.append("subject_id = %s")
        params.append(int(subject_id))
    if difficulty:
        conditions.append("difficulty = %s")
        params.append(difficulty)
    where = " AND ".join(conditions)
    cur.execute(f"SELECT * FROM questions WHERE {where} ORDER BY RANDOM() LIMIT %s", params + [limit])
    qs = dict_fetchall(cur)
    for q in qs:
        q.pop('correct_option', None)
        q.pop('explanation', None)
    cur.close(); db.close()
    return jsonify(qs)


@app.route('/api/mock/daily')
def daily_mock_test():
    section_map = {1:('Quantitative Aptitude',25), 2:('General Intelligence & Reasoning',25), 3:('English Language',25), 4:('General Awareness',25)}
    db = get_db()
    cur = db.cursor()
    sections = []
    for subject_id, (subject_name, count) in section_map.items():
        cur.execute("SELECT id,question_text,option_a,option_b,option_c,option_d,difficulty FROM questions WHERE subject_id=%s ORDER BY RANDOM() LIMIT %s", (subject_id, count))
        qs = dict_fetchall(cur)
        sections.append({'subject_id': subject_id, 'subject_name': subject_name, 'questions': qs, 'count': len(qs)})
    cur.close(); db.close()
    total = sum(s['count'] for s in sections)
    return jsonify({'date': str(date.today()), 'total': total, 'duration': 3600, 'sections': sections})


@app.route('/api/leaderboard')
def leaderboard():
    lb_type = request.args.get('type', 'score')
    subject_id = request.args.get('subject_id')
    limit = min(int(request.args.get('limit', 10)), 50)
    db = get_db()
    cur = db.cursor()
    base = """
        SELECT qa.id, qa.subject_id, s.name as subject_name, s.icon,
               qa.total_questions, qa.correct_answers, qa.wrong_answers,
               qa.score_percent, qa.time_taken_seconds, qa.attempted_at
        FROM quiz_attempts qa
        LEFT JOIN subjects s ON qa.subject_id = s.id
        WHERE qa.total_questions > 0
    """
    params = []
    if subject_id:
        base += " AND qa.subject_id = %s"
        params.append(int(subject_id))
    if lb_type == 'speed':
        base += " AND qa.score_percent >= 60 ORDER BY qa.time_taken_seconds ASC"
    else:
        base += " ORDER BY qa.score_percent DESC, qa.correct_answers DESC"
    base += " LIMIT %s"
    params.append(limit)
    cur.execute(base, params)
    rows = dict_fetchall(cur)
    for i, r in enumerate(rows):
        r['rank'] = i + 1
        r['score_percent'] = float(r['score_percent'] or 0)
        if r.get('attempted_at'):
            r['attempted_at'] = str(r['attempted_at'])
    cur.close(); db.close()
    return jsonify({'type': lb_type, 'entries': rows})


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
        ORDER BY avg_score DESC
    """)
    subject_stats = dict_fetchall(cur)
    for s in subject_stats:
        s['avg_score'] = round(float(s['avg_score'] or 0), 1)
        s['best_score'] = round(float(s['best_score'] or 0), 1)
        s['accuracy'] = round(float(s['total_correct'] or 0) / float(s['total_answered'] or 1) * 100, 1)
    weak_subjects = [s['name'] for s in subject_stats if s['accuracy'] < 50]
    strong_subjects = [s['name'] for s in subject_stats if s['accuracy'] >= 75]
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
    cur.execute("SELECT COUNT(*) as total_attempts, AVG(score_percent) as overall_avg, SUM(correct_answers) as total_correct, SUM(wrong_answers) as total_wrong, SUM(skipped) as total_skipped FROM quiz_attempts")
    overall = dict_fetchone(cur)
    overall['overall_avg'] = round(float(overall['overall_avg'] or 0), 1)
    cur.execute("""
        SELECT q.id, q.question_text, q.subject_id, s.name as subject_name, COUNT(*) as wrong_count
        FROM attempt_details ad
        JOIN questions q ON ad.question_id = q.id
        LEFT JOIN subjects s ON q.subject_id = s.id
        WHERE ad.is_correct = 0
        GROUP BY q.id, q.question_text, q.subject_id, s.name
        HAVING COUNT(*) >= 2
        ORDER BY wrong_count DESC LIMIT 10
    """)
    weak_questions = dict_fetchall(cur)
    cur.close(); db.close()
    return jsonify({'overall': overall, 'subject_stats': subject_stats, 'weak_subjects': weak_subjects, 'strong_subjects': strong_subjects, 'score_trend': trend, 'weak_questions': weak_questions})


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
    required_cols = {'question_text','option_a','option_b','option_c','option_d','correct_option'}
    if not required_cols.issubset(set(reader.fieldnames or [])):
        missing = required_cols - set(reader.fieldnames or [])
        return jsonify({'error': f'Missing CSV columns: {missing}'}), 400
    db = get_db()
    cur = db.cursor()
    inserted = skipped = 0
    for row in reader:
        co = row.get('correct_option','').strip().upper()
        if co not in ('A','B','C','D'):
            skipped += 1; continue
        diff = row.get('difficulty','').strip().lower()
        if diff not in ('easy','medium','hard'):
            diff = predict_difficulty(row['question_text'], row.get('explanation',''))
        try:
            cur.execute("INSERT INTO questions (subject_id,question_text,option_a,option_b,option_c,option_d,correct_option,explanation,difficulty) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (int(subject_id), row['question_text'].strip(), row['option_a'].strip(), row['option_b'].strip(), row['option_c'].strip(), row['option_d'].strip(), co, row.get('explanation','').strip(), diff))
            inserted += 1
        except Exception as e:
            skipped += 1
    db.commit(); cur.close(); db.close()
    return jsonify({'inserted': inserted, 'skipped': skipped}), 201


@app.route('/api/ai/explain', methods=['POST'])
def ai_generate_explanation():
    data = request.get_json(silent=True) or {}
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
        if q.get('explanation','').strip():
            cur.close(); db.close()
            return jsonify({'explanation': q['explanation'], 'cached': True})
    else:
        q = data
    required = ['question_text','option_a','option_b','option_c','option_d','correct_option']
    if not all(q.get(k) for k in required):
        return jsonify({'error': 'Provide question_id OR full question fields'}), 400
    prompt = (f"Question: {q['question_text']}\nA) {q['option_a']}  B) {q['option_b']}  C) {q['option_c']}  D) {q['option_d']}\nCorrect Answer: {q['correct_option']}\n\nWrite a clear, concise explanation (2-4 sentences) why this answer is correct.")
    try:
        explanation = call_groq("You are an SSC CGL expert tutor.", prompt, max_tokens=400)
    except Exception as e:
        return jsonify({'error': 'AI service error', 'detail': str(e)}), 502
    if qid and db:
        cur.execute("UPDATE questions SET explanation=%s WHERE id=%s", (explanation, int(qid)))
        db.commit(); cur.close(); db.close()
    return jsonify({'explanation': explanation, 'cached': False})


@app.route('/api/ai/tutor', methods=['POST'])
def ai_tutor():
    data = request.get_json(silent=True) or {}
    message = data.get('message', '').strip()
    history = data.get('history', [])
    if not message:
        return jsonify({'error': 'message is required'}), 400
    system_prompt = (
        "You are an expert SSC CGL 2026 tutor helping a student named Samin prepare for the exam.\n"
        "Specialise in: Quantitative Aptitude, General Intelligence & Reasoning, English Language, General Awareness.\n"
        "Give step-by-step solutions, use simple language, be encouraging and exam-focused."
    )
    messages = []
    for h in history[-10:]:
        role = h.get('role','')
        content = h.get('content','')
        if role in ('user','assistant') and content:
            messages.append({'role': role, 'content': content})
    messages.append({'role': 'user', 'content': message})
    try:
        reply = call_claude(system_prompt, messages, max_tokens=1024)
    except Exception as e:
        return jsonify({'error': 'AI service error', 'detail': str(e)}), 502
    return jsonify({'reply': reply})


@app.route('/api/ai/study-plan', methods=['POST'])
def ai_study_plan():
    data = request.get_json(silent=True) or {}
    exam_date = data.get('exam_date', 'September 2026')
    hours_per_day = int(data.get('hours_per_day', 3))
    weak_subjects = data.get('weak_subjects', [])
    weak_str = ', '.join(weak_subjects) if weak_subjects else 'None identified yet'
    user_prompt = (f"Create a 4-week SSC CGL 2026 study plan. Exam: {exam_date}. Hours/day: {hours_per_day}. Weak subjects: {weak_str}. Include weekly breakdown, daily schedule, top 5 tips, and final week revision strategy.")
    try:
        plan = call_claude("You are an expert SSC CGL exam coach.", [{'role':'user','content':user_prompt}], max_tokens=1500)
    except Exception as e:
        return jsonify({'error': 'AI service error', 'detail': str(e)}), 502
    return jsonify({'plan': plan, 'weak_subjects': weak_subjects})


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
    prompt = (f"Question: {q['question_text']}\nA) {q.get('option_a','')}  B) {q.get('option_b','')}  C) {q.get('option_c','')}  D) {q.get('option_d','')}\n\nGive ONE helpful hint without revealing the answer. 1-2 sentences.")
    try:
        hint = call_groq("You are an SSC CGL tutor giving helpful hints.", prompt, max_tokens=200)
    except Exception as e:
        return jsonify({'error': 'AI service error', 'detail': str(e)}), 502
    return jsonify({'hint': hint})


@app.route('/api/ai/status')
def ai_status():
    groq_ok = bool(GROQ_API_KEY)
    anthropic_ok = bool(ANTHROPIC_API_KEY)
    tutor_provider = 'claude' if anthropic_ok else ('groq' if groq_ok else 'none')
    return jsonify({'groq': {'configured': groq_ok, 'model': GROQ_MODEL}, 'claude': {'configured': anthropic_ok}, 'tutor_provider': tutor_provider, 'all_features_active': groq_ok})


@app.errorhandler(404)
def not_found(e):
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Not found'}), 404
    return send_from_directory('static', 'index.html')


@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500


# Init DB at module load (works with gunicorn)
try:
    init_db()
except Exception as e:
    app.logger.warning(f"DB init failed: {e}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
