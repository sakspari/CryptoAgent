import schedule
import time
import pytz
from datetime import datetime
from src.main import main as run_agent
from src.train_model import train as run_training

# Timezone
JAKARTA_TZ = pytz.timezone('Asia/Jakarta')

def job_prediction():
    print(f"\n[Scheduler] Starting Prediction Job at {datetime.now(JAKARTA_TZ)}")
    try:
        run_agent()
        print(f"[Scheduler] Prediction Job finished at {datetime.now(JAKARTA_TZ)}")
    except Exception as e:
        print(f"[Scheduler] Prediction Job failed: {e}")

def job_training():
    print(f"\n[Scheduler] Starting Monthly Training Job at {datetime.now(JAKARTA_TZ)}")
    try:
        run_training()
        print(f"[Scheduler] Training Job finished at {datetime.now(JAKARTA_TZ)}")
    except Exception as e:
        print(f"[Scheduler] Training Job failed: {e}")

def start_scheduler():
    print(f"[Scheduler] Service started. Timezone: {JAKARTA_TZ}")
    print("[Scheduler] Scheduled for 07:00 and 14:00 daily (Prediction).")
    print("[Scheduler] Scheduled for 1st of month at 02:00 (Training).")
    
    while True:
        now_jakarta = datetime.now(JAKARTA_TZ)
        current_time = now_jakarta.strftime("%H:%M")
        day_of_month = now_jakarta.day
        
        # Daily Prediction
        if current_time == "07:00":
            job_prediction()
            time.sleep(61)
        elif current_time == "14:00":
            job_prediction()
            time.sleep(61)
            
        # Monthly Training (1st day of month at 02:00 AM)
        if day_of_month == 1 and current_time == "02:00":
            job_training()
            time.sleep(61)
            
        time.sleep(30)

if __name__ == "__main__":
    start_scheduler()
