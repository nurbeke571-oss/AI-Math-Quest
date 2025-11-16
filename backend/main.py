import os
import time
import random
import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.responses import FileResponse
import numpy as np
from sklearn.linear_model import LogisticRegression
from typing import List, Tuple, Optional

# ==================== ДЕРЕКҚОР КОНФИГУРАЦИЯСЫ ======================
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base

# 🛑 Render-де DATABASE_URL айнымалысын пайдаланамыз
# Локальді тексеру үшін SQLite-тің әдепкі мәнін қолдануға болады
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./math_quest.db")

# SQLAlchemy Engine және Session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Дерекқор тәуелділігін алу функциясы
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 1. Player Дерекқор Моделі
class PlayerDB(Base):
    """Ойыншы деректерін сақтауға арналған SQLAlchemy моделі."""
    __tablename__ = "players"

    player = Column(String, primary_key=True, index=True)
    level = Column(Integer, default=1)
    score = Column(Integer, default=0)
    # current_answer және asked_questions деректерді тұрақты сақтау үшін
    current_answer = Column(Float, nullable=True) 
    asked_questions = Column(String, default="") # Сұрақтарды жол ретінде сақтау үшін

# Кестені құру (Ескерту: Render-де кестенің құрылғанына көз жеткізіңіз)
Base.metadata.create_all(bind=engine)

# ==================== API Инициализация ======================
app = FastAPI(title="AI Math Quest — PostgreSQL Version")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Static Files (Frontend)
frontend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
sounds_path = os.path.join(frontend_path, "sounds")

if os.path.exists(sounds_path):
    app.mount("/sounds", StaticFiles(directory=sounds_path), name="sounds")

# ==================== Ойын деректері ======================
# players сөздігі жойылды!

# ... (tasks сөздігі бұрынғыдай қалады) ...
# Сондықтан tasks сөздігін осы жерге қоюдың қажеті жоқ, ол алдыңғы кодтан алынады.

tasks = {
    1: [  # Уровень 1 — Простая арифметика
        ("2 + 3", 5), ("4 + 5", 9), ("7 - 2", 5), ("9 - 3", 6),
        ("1 + 6", 7), ("8 - 4", 4), ("5 + 3", 8), ("6 - 1", 5),
        ("10 - 7", 3), ("3 + 4", 7), ("9 - 8", 1), ("2 + 2", 4),
        ("8 - 5", 3), ("5 + 4", 9), ("6 + 3", 9), ("7 - 6", 1),
        ("4 + 6", 10), ("3 + 3", 6), ("9 - 5", 4), ("10 - 2", 8)
    ],
    2: [  # Уровень 2 — Умножение и деление
        ("2 * 3", 6), ("4 * 2", 8), ("9 / 3", 3), ("12 / 4", 3),
        ("5 * 5", 25), ("8 / 2", 4), ("7 * 3", 21), ("6 * 6", 36),
        ("15 / 5", 3), ("10 / 2", 5), ("9 * 9", 81), ("3 * 4", 12),
        ("18 / 3", 6), ("14 / 2", 7), ("11 * 2", 22), ("16 / 4", 4),
        ("5 * 7", 35), ("20 / 5", 4), ("8 * 3", 24), ("24 / 6", 4)
    ],
    3: [  # Уровень 3 — Степени и корни
        ("2^3", 8), ("3^2", 9), ("4^2", 16), ("5^2", 25),
        ("√9", 3), ("√16", 4), ("√25", 5), ("√36", 6),
        ("2^4", 16), ("3^3", 27), ("4^3", 64), ("5^3", 125),
        ("√49", 7), ("√81", 9), ("2^5", 32), ("√100", 10),
        ("3^4", 81), ("√64", 8), ("√121", 11), ("2^6", 64)
    ],
    4: [  # Уровень 4 — Смешанные операции
        ("3 + 4 * 2", 11), ("8 / 2 + 5", 9), ("6 - 3 + 2", 5),
        ("9 - 2 * 3", 3), ("10 / 5 + 6", 8), ("4 + 6 / 2", 7),
        ("7 + 3 * 2", 13), ("8 - 2 + 5", 11), ("12 / 3 + 4", 8),
        ("2 * 3 + 4", 10), ("5 + 10 / 2", 10), ("9 - 3 + 6", 12),
        ("8 / 4 + 7", 9), ("10 - 2 * 4", 2), ("6 + 2 * 3", 12),
        ("7 * 2 - 5", 9), ("9 / 3 + 8", 11), ("8 + 6 / 2", 11),
        ("12 / 4 + 9", 12), ("3 * 3 + 1", 10)
    ],
    5: [  # Уровень 5 — Логарифмы
        ("log_2(8)", 3), ("log_10(1000)", 3), ("log_3(27)", 3),
        ("log_5(25)", 2), ("log_2(16)", 4), ("log_4(64)", 3),
        ("log_10(100)", 2), ("log_2(32)", 5), ("log_3(81)", 4),
        ("log_6(36)", 2), ("log_7(49)", 2), ("log_2(4)", 2),
        ("log_9(81)", 2), ("log_10(10000)", 4), ("log_8(64)", 2),
        ("log_3(9)", 2), ("log_5(125)", 3), ("log_2(128)", 7),
        ("log_4(256)", 4), ("log_2(1024)", 10)
    ],
    6: [  # Уровень 6 — Тригонометрия
        ("sin(30°)", 0.5), ("cos(60°)", 0.5), ("tan(45°)", 1),
        ("sin(90°)", 1), ("cos(0°)", 1), ("sin(0°)", 0),
        ("cos(90°)", 0), ("tan(0°)", 0), ("sin(45°)", 0.7071),
        ("cos(45°)", 0.7071), ("tan(30°)", 0.5774), ("sin(60°)", 0.866),
        ("cos(30°)", 0.866), ("tan(60°)", 1.732), ("sin(180°)", 0),
        ("cos(180°)", -1), ("sin(270°)", -1), ("cos(270°)", 0),
        ("tan(90°)", None), ("sin(120°)", 0.866)
    ],
    7: [  # Уровень 7 — Дроби и проценты
        ("1/2 + 1/4", 0.75), ("3/5 + 2/5", 1.0), ("1/3 + 1/6", 0.5),
        ("50% от 200", 100), ("25% от 80", 20), ("10% от 500", 50),
        ("3/4 - 1/2", 0.25), ("2/3 + 1/3", 1), ("20% от 400", 80),
        ("5/10 + 2/10", 0.7), ("75% от 120", 90), ("1/5 от 50", 10),
-      ("2/5 + 1/5", 0.6), ("30% от 300", 90), ("10% от 250", 25),
        ("40% от 150", 60), ("3/8 + 1/8", 0.5), ("60% от 90", 54),
        ("25% от 160", 40), ("80% от 50", 40)
    ],
    8: [  # Уровень 8 — Отрицательные числа
        ("-3 + 7", 4), ("5 - 9", -4), ("-4 + -6", -10),
        ("-10 + 15", 5), ("8 - -3", 11), ("-7 - 2", -9),
        ("-5 + 9", 4), ("10 - -5", 15), ("-2 * 3", -6),
        ("-8 / 2", -4), ("-3 - -3", 0), ("-12 / -3", 4),
        ("-4 * -2", 8), ("-15 + 10", -5), ("-1 * 7", -7),
        ("8 + -10", -2), ("-9 / 3", -3), ("-5 + 2", -3),
        ("-6 - 4", -10), ("-2 * -5", 10)
    ],
    9: [  # Уровень 9 — Комбинированные выражения
        ("(3 + 5) * 2", 16), ("(8 - 4) / 2", 2), ("(6 + 2) * (3 - 1)", 16),
        ("(10 / 2) + (3 * 2)", 11), ("(9 - 3) * (2 + 1)", 18),
        ("(4 + 6) / (2 + 3)", 2), ("(8 / 2) * (3 + 1)", 16),
        ("(5 + 3) * (2 + 2)", 32), ("(12 - 6) / 3", 2),
        ("(9 - 3) * 2", 12), ("(15 / 3) + (4 * 2)", 13),
        ("(18 / 6) + (5 * 3)", 17), ("(10 - 5) * 4", 20),
        ("(7 + 3) / 2", 5), ("(6 + 9) / 3", 5),
        ("(8 / 2) + 7", 11), ("(9 / 3) * 2", 6),
        ("(12 / 4) * 3", 9), ("(15 - 9) * 2", 12), ("(8 + 4) / 2", 6)],
    10: [  # Уровень 10 — Сложные функции
        ("2^3 + √49", 15), ("log_10(1000) + 3^2", 12), ("sin(30°) * 10", 5),
        ("cos(60°) * 10", 5), ("tan(45°) + 2^3", 9), ("log_2(16) + √25", 9),
        ("3^3 - 2^2", 23), ("√81 + log_3(27)", 12), ("sin(90°) + cos(0°)", 2),
        ("tan(30°) * 10", 5.774), ("log_2(8) + 4^2", 19), ("√64 + 2^3", 16),
        ("3^2 + 4^2", 25), ("√100 - log_10(100)", 8), ("sin(60°)*10", 8.66),
        ("cos(30°)*10", 8.66), ("tan(60°)*5", 8.66), ("log_5(25)+√36", 8),
        ("2^5 - √49", 25), ("log_2(32)+sin(90°)", 6)]}

def generate_question(level: int, asked_questions_str: str) -> Tuple[str, float, str]:
    """Жаңа сұрақ генерациялайды және қолданылған сұрақтарды жаңартады."""
    asked = asked_questions_str.split('|') if asked_questions_str else []
    
    if level not in tasks:
        level = 1
        
    available = [q for q in tasks[level] if q[0] not in asked]
    
    if not available:
        asked.clear()
        available = tasks[level]
        
    question, answer = random.choice(available)
    asked.append(question)
    
    new_asked_str = "|".join(asked)
    
    if answer is None:
        answer = 0.0
        
    return question, round(answer, 4), new_asked_str


# ==================== AI Үлгісі (Дерекқорсыз) ======================
# AI үлгісін әлі де жаһандық айнымалы ретінде қалдырамыз, 
# бірақ оның деректерін сақтау үшін DB-ге көшу қажет болуы мүмкін.
ai_model = LogisticRegression()
training_data: List[List[float]] = []
labels: List[int] = []

def train_ai_model():
    # ... (бұрынғыдай қалады) ...
    global ai_model
    if len(training_data) > 5:
        X = np.array(training_data)
        y = np.array(labels)
        ai_model.fit(X, y)
        print("🧠 AI үлгі үйретілді (обучена)")

def predict_next_level(score, level):
    # ... (бұрынғыдай қалады) ...
    if len(training_data) < 5:
        return 0.5
    X_pred = np.array([[score, level]])
    try:
        prob = ai_model.predict_proba(X_pred)[0][1]
    except Exception:
        prob = 0.5
    return float(prob)

def adaptive_difficulty(prob, current_level):
    # ... (бұрынғыдай қалады) ...
    MAX_LEVEL = 10 
    if prob > 0.75:
        new_level = current_level + 1
    elif prob < 0.4:
        new_level = max(1, current_level - 1)
    else:
        new_level = current_level
    return min(new_level, MAX_LEVEL)


# ==================== API (DB CRUD) ======================

class AnswerRequest(BaseModel):
    player: str
    question: str
    user_answer: str

# 1. Тіркеу (Register)
@app.get("/register/{player}")
def register_player_route(player: str, db: Session = Depends(get_db)):
    """Ойыншыны тіркейді немесе бұрынғы деректерді қайтарады."""
    db_player = db.query(PlayerDB).filter(PlayerDB.player == player).first()

    if not db_player:
        db_player = PlayerDB(player=player, level=1, score=0, current_answer=None, asked_questions="")
        db.add(db_player)
        db.commit()
        db.refresh(db_player)
        print(f"🧍‍♂️ Ойыншы тіркелді (DB): {player}")
    
    return {"player": db_player.player, 
            "current_score": db_player.score, 
            "current_level": db_player.level}


# 2. Сұрақ алу (Get Question)
@app.get("/question/{player}")
def get_question(player: str, db: Session = Depends(get_db)):
    db_player = db.query(PlayerDB).filter(PlayerDB.player == player).first()

    if not db_player:
        # Егер ойыншы табылмаса (өте сирек болуы керек), оны қайта тіркеу
        raise HTTPException(status_code=404, detail="Ойыншы табылмады. Қайта тіркеліңіз.")
    
    # AI Адаптация
    prob = predict_next_level(db_player.score, db_player.level)
    new_level = adaptive_difficulty(prob, db_player.level)
    
    # Сұрақ генерациясы
    q, ans, new_asked_str = generate_question(new_level, db_player.asked_questions)
    
    # Деректерді жаңарту және сақтау
    db_player.level = new_level
    db_player.current_answer = ans
    db_player.asked_questions = new_asked_str
    
    db.commit()
    
    return {"math_question": q, 
            "current_level": db_player.level, 
            "ai_prediction": round(prob, 2)}


# 3. Жауап беру (Answer)
@app.post("/answer")
def answer(req: AnswerRequest, db: Session = Depends(get_db)):
    db_player = db.query(PlayerDB).filter(PlayerDB.player == req.player).first()
    
    if not db_player:
        raise HTTPException(status_code=404, detail="Ойыншы табылмады.")

    correct = db_player.current_answer
    if correct is None:
        raise HTTPException(status_code=400, detail="Сначала получите вопрос.")

    # user_answer-ді str-ден float-қа қолмен түрлендіреміз
    try:
        user_answer = float(req.user_answer)
    except ValueError:
        return {"is_correct": False, "message": "Жауап сан болуы керек!", "new_level": db_player.level, "score": db_player.score}

    is_correct = abs(user_answer - correct) < 0.01

    if is_correct:
        db_player.score += 10 * db_player.level
    else:
        db_player.score = max(db_player.score - 5, 0)
    
    # current_answer мәнін тазалау
    db_player.current_answer = None

    # === AI оқыту ===
    training_data.append([db_player.score, db_player.level])
    labels.append(1 if is_correct else 0)
    train_ai_model()

    # AI келесі деңгей туралы шешім қабылдайды
    prob = predict_next_level(db_player.score, db_player.level)
    db_player.level = adaptive_difficulty(prob, db_player.level)

    # Дерекқорға өзгерістерді сақтау
    db.commit()
    db.refresh(db_player)

    progress = int((db_player.level / 10) * 100)
    print(f"[{req.player}] {'✅' if is_correct else '❌'} {req.question} — жаңа деңгей {db_player.level} | Болжау: {prob:.2f}")
    
    return {
        "is_correct": is_correct,
        "correct_answer": correct,
        "new_level": db_player.level,
        "score": db_player.score,
        "progress": progress,
        "ai_prediction": round(prob, 2)
    }

# 4. Көшбасшылар тақтасы (Leaderboard)
@app.get("/leaderboard")
def leaderboard(db: Session = Depends(get_db)):
    sorted_players = db.query(PlayerDB).order_by(PlayerDB.score.desc()).limit(10).all()
    return [{"player": p.player, "score": p.score} for p in sorted_players]

# --------------------------------------------------------
# Static Files (ең соңында)
# --------------------------------------------------------
app.mount("/", StaticFiles(directory=frontend_path, html=True), name="static")

# ==================== Іске қосу ======================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print("🎮 Қазақша AI Math Quest сервері іске қосылды (DB persistence)...")
    uvicorn.run(app, host="0.0.0.0", port=port)
