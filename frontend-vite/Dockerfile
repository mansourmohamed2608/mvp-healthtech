# === Stage 1: build Vite app ===
FROM node:22-alpine AS build

WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm install

# Copy source
COPY . .

# Build for production
RUN npm run build

# === Stage 2: serve static files with nginx ===
FROM nginx:alpine

# Copy built files from previous stage
COPY --from=build /app/dist /usr/share/nginx/html

# Expose HTTP port inside the container
EXPOSE 80

# Run nginx
CMD ["nginx", "-g", "daemon off;"]
