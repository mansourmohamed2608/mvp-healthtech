FROM node:22-alpine
RUN apk add --no-cache python3 make g++
RUN corepack enable && corepack prepare pnpm@latest --activate
WORKDIR /app
COPY package.json pnpm-lock.yaml* ./
RUN pnpm install
RUN pnpm rebuild bcrypt
COPY tsconfig*.json ./
COPY src ./src
EXPOSE 3000
CMD ["pnpm","start:dev"]
