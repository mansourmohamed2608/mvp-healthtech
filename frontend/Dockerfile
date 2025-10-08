FROM node:22-alpine
RUN corepack enable && corepack prepare pnpm@latest --activate
WORKDIR /app
COPY package.json pnpm-lock.yaml* ./
RUN pnpm install || true
COPY next.config.* ./
COPY tsconfig*.json ./
COPY src ./src
COPY public ./public
EXPOSE 3000
CMD ["pnpm","dev","--port","3000","--hostname","0.0.0.0"]
