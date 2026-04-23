# Stage 1: Build Phase
FROM node:20-alpine AS builder

WORKDIR /app

# Install dependencies first for caching module layer
COPY package.json package-lock.json* ./
RUN npm install

# Copy source code and build frontend
COPY . .
RUN npm run build

# Stage 2: Production Serving Phase (Nginx)
FROM nginx:alpine

# Remove default nginx static files
RUN rm -rf /usr/share/nginx/html/*

# Copy compiled artifacts from Stage 1
COPY --from=builder /app/dist /usr/share/nginx/html

# Expose HTTP port
EXPOSE 80

# Start Nginx server
CMD ["nginx", "-g", "daemon off;"]