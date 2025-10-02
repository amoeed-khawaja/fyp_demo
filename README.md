# Mall Profiling Demo (Prototype)

This is a proof-of-concept demo for a mall profiling system.  
It uses **face recognition** to assign unique IDs to new visitors,  
recognizes returning visitors, logs their visits, and shows a mock ad recommendation.

## 🔧 Setup

1. Clone repo / extract files:
   ```bash
   git clone <repo-url>
   cd mall-profiling-demo
   ```

2. Create virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate    # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the demo:
   ```bash
   python main.py
   ```

## 🎥 Usage

- The webcam opens and detects faces.
- New faces get a unique ID (`C1`, `C2`, …).
- Returning faces are recognized automatically.
- Each visit is logged in `database.json`.
- A sample **ad recommendation** is displayed on screen.

Press **`q`** to exit.
