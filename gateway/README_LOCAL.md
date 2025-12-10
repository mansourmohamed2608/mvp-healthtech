# Local Gateway (Nest) without Docker

Prereqs: Node 18+, npm. Postgres + Redis running locally (see root LOCAL_DEV.md).

1) Install
```
cd gateway
npm install
```

2) Env
```
cp .env.local.example .env.local
```
Edit DB/Redis URLs if needed. Ensure INTERNAL_SECRET/JWT_SECRET match orchestrator/llm-va.

3) Run
```
npm run start:dev
```

Gateway listens on the port from `.env.local` (default 3001).
