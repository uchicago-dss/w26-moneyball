git clone https://github.com/uchicago-dss/w26-moneyball.git
cd w26-moneyball
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python models/wc_model.py
