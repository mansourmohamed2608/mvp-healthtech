#!/bin/bash
# Generate local development secrets
# This script creates a .env file with randomly generated secrets

set -e

ENV_FILE=".env"
ENV_EXAMPLE=".env.example"

if [ -f "$ENV_FILE" ]; then
    echo "⚠️  .env file already exists"
    read -p "Overwrite? (y/N): " confirm
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        echo "Aborted."
        exit 0
    fi
fi

# Generate random secrets
generate_secret() {
    openssl rand -hex 32
}

generate_short_secret() {
    openssl rand -hex 16
}

echo "🔐 Generating local development secrets..."

# Start with example as base
cp "$ENV_EXAMPLE" "$ENV_FILE"

# Generate and replace secrets
JWT_SECRET=$(generate_secret)
INTERNAL_SECRET=$(generate_secret)
WS_SHARED_SECRET=$(generate_short_secret)

# Replace placeholders with generated values
sed -i "s/^JWT_SECRET=.*/JWT_SECRET=$JWT_SECRET/" "$ENV_FILE"
sed -i "s/^INTERNAL_SECRET=.*/INTERNAL_SECRET=$INTERNAL_SECRET/" "$ENV_FILE"
sed -i "s/^WS_SHARED_SECRET=.*/WS_SHARED_SECRET=$WS_SHARED_SECRET/" "$ENV_FILE"

# Set default local database URL
sed -i "s|^DATABASE_URL=.*|DATABASE_URL=postgresql://postgres:postgres@localhost:5432/healthtech|" "$ENV_FILE"

echo "✅ Generated secrets in $ENV_FILE"
echo ""
echo "⚠️  Remember:"
echo "   - Never commit .env to version control"
echo "   - These secrets are for LOCAL DEVELOPMENT ONLY"
echo "   - Production secrets must come from a secrets manager"
echo ""
echo "Generated secrets:"
echo "   JWT_SECRET: ${JWT_SECRET:0:10}..."
echo "   INTERNAL_SECRET: ${INTERNAL_SECRET:0:10}..."
echo "   WS_SHARED_SECRET: ${WS_SHARED_SECRET:0:10}..."
