# === Stage 1: build Vite app ===
FROM node:22-alpine AS build

WORKDIR /app

# Build-time config injected by docker compose build args
ARG VITE_API_URL
ARG VITE_OIDC_ENABLED=false
ENV VITE_API_URL=$VITE_API_URL
ENV VITE_OIDC_ENABLED=$VITE_OIDC_ENABLED

# Install dependencies
COPY package*.json ./
RUN npm install

# Copy source
COPY . .

# Build for production
RUN npm run build

# === Stage 2: serve static files with nginx ===
FROM nginx:alpine

# Copy custom nginx config for SPA routing
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy built files from previous stage
COPY --from=build /app/dist /usr/share/nginx/html

# Expose HTTP port inside the container
EXPOSE 80

# Run nginx
CMD ["nginx", "-g", "daemon off;"]
