# Vitalis React UI

A calm, sage-toned React + Tailwind frontend for the AI healthcare project.

## What It Includes

- Professional multi-screen healthcare dashboard
- Searchable symptom selection powered by the Flask API
- Live disease recommendation result screen
- Medication, diet, exercise, precautions, and care-plan views
- Confidence, reliability, warnings, and top prediction visualization
- Fully responsive layout for desktop and mobile
- Framer Motion animations and Lucide medical icons

## API Setup

Start the Flask API from the sibling `Medical Chatbot` folder:

```bash
cd "../Medical Chatbot"
python flask_api.py
```

The UI uses `http://localhost:5000` by default. To point at a different API:

```bash
VITE_API_BASE_URL=http://localhost:5000 npm run dev
```

## Run Locally

```bash
cd healthcare-ui
npm install
npm run dev
```

## Build

```bash
npm run build
```

## Notes

The recommendation flow is integrated with the Python API. If the API is not
running, the UI still renders with built-in example symptoms and shows a clear
connection notice.

## Deploy (Vercel)

This is a static Vite build, so only this `healthcare-ui/` folder deploys to
Vercel — the Flask/RAG backend needs a separate host that can run a
long-lived Python process (Render, Railway, Fly.io, a VPS, etc.), since it
loads a local embedding model and FAISS index that don't fit Vercel's
serverless model.

1. In the Vercel dashboard, import this repo and set **Root Directory** to
   `healthcare-ui`. Framework preset `Vite` and the build command
   `npm run build` / output `dist` are auto-detected.
2. Add an environment variable `VITE_API_BASE_URL` pointing at wherever
   `flask_api.py` ends up hosted (e.g. `https://your-backend.onrender.com`).
   Vite bakes this in at build time, so redeploy after changing it.
3. On the backend, add the Vercel domain to `ALLOWED_ORIGIN`/`ALLOWED_ORIGINS`
   so `flask_api.py`'s CORS allowlist accepts requests from it.
