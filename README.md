# DisasterVision Frontend

A **Next.js 14** frontend for DisasterVision – AI‑powered satellite image analysis for landslide & flood prediction.

## Features
- Home, Prediction, About Us, Contact, Register & Login pages
- Fetches Sentinel Hub satellite imagery and WeatherAPI weather data
- Tailwind CSS styling + Framer Motion animations
- Cascading Country → State → City dropdown selector
- Fully TypeScript, App Router

## Quick Start

```bash
git clone https://github.com/your-org/disastervision-frontend.git
cd disastervision-frontend
pnpm i # or npm install / yarn
cp .env.example .env.local
# add your API keys in .env.local
pnpm dev
```

Navigate to **http://localhost:3000**.
