
**Python Backend** -- RL-based dispatcher + delay prediction -
**Node.js Bridge** -- WebSocket bridge connecting backend to frontend -
**React + Konva Frontend** -- Real-time simulation, train animations &
AI suggestions

------------------------------------------------------------------------

Project Structure

    rail/
    ├── backend/              # Python backend (state manager, websocket server)
    │   ├── state_manager.py
    │   ├── websocket_server.py
    │   └── ...
    ├── model/                # RL & delay prediction models
    │   └── dispatcher2.py
    ├── frontend/             # React + TypeScript frontend
    │   ├── src/
    │   │   ├── components/   # Train.tsx, Simulation.tsx, SuggestionsPanel.tsx, etc.
    │   │   └── ...
    │   └── server.ts         # WebSocket bridge (Node.js)
    └── README.md

------------------------------------------------------------------------

Quick Setup

Backend (Python)

**Requirements:** Python 3.10+, `venv`

``` bash
cd backend
python -m venv .venv
source .venv/Scripts/activate  # (or .venv/bin/activate on macOS/Linux)

pip install -r requirements.txt
python websocket_server.py
```

This will start the **backend WebSocket server** on
`ws://localhost:8765`.


Bridge (Node.js)

**Requirements:** Node.js 18+, npm

``` bash
cd frontend
npm install
npm run server
```

This runs the **WebSocket bridge** on `ws://localhost:3001` that relays
data between Python backend & frontend.


Frontend (React + Vite)

In a **second terminal**:

``` bash
cd frontend
npm run dev
```

This starts the React app on `http://localhost:5173`.



Demo Flow

1.  Start **backend** (`python websocket_server.py`)
2.  Start **bridge** (`npm run server`)
3.  Start **frontend** (`npm run dev`)
4.  Open <http://localhost:5173> and watch trains + suggestions update
    live.


Current Features

✅ Real-time train simulation (React-Konva)\
✅ RL-based delay mitigation model (PPO)\
✅ WebSocket bridge for backend-frontend sync\
✅ AI suggestions panel (accept/reject actions)\
✅ Modular architecture -- ready for further UI polish


Next Steps (For Frontend )

-   Polish **train animation curves & speed scaling**
-   Add **signal color transitions** (red → green)
-   Improve **UI/UX** for suggestions panel (click-to-apply)
-   Visualize **metrics dashboard** (on-time %, conflicts prevented)
-   

Contributing

Focus areas: - Better visualization of track network 
More intuitive UI/UX for train interactions 
Styling & theming 

Demo Tips

When demoing: - Keep **backend + bridge + frontend** running
simultaneously. - Open browser console to show incoming WS data
(`state_update` & `suggestions`). - Highlight AI suggestions being
applied live.

