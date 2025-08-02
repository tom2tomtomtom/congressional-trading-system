#!/bin/bash
# Congressional Trading Intelligence System - Railway CLI Deployment Script

set -e  # Exit on error

echo "🚀 Congressional Trading Intelligence System - Railway Deployment"
echo "=================================================================="

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not found. Installing..."
    npm install -g @railway/cli
    echo "✅ Railway CLI installed"
fi

# Login check
echo "🔑 Checking Railway authentication..."
if ! railway whoami &> /dev/null; then
    echo "Please login to Railway:"
    railway login
fi

# Initialize project if needed
if [ ! -f "railway.toml" ]; then
    echo "📦 Initializing Railway project..."
    railway init
fi

echo "⚙️  Setting up environment variables..."

# Set API keys
echo "Setting API keys..."
railway variables set CONGRESS_GOV_API_KEY=GnEJVyPiswjccfl3Y9KHhwRVEeWDUnxOAVC4aMhD
railway variables set FINNHUB_API_KEY=d26nr7hr01qvrairb710d26nr7hr01qvrairb71g

# Generate and set security keys
echo "Generating security keys..."
SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
JWT_SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")

railway variables set SECRET_KEY="$SECRET_KEY"
railway variables set JWT_SECRET_KEY="$JWT_SECRET_KEY"

# Application settings
echo "Setting application configuration..."
railway variables set FLASK_ENV=production
railway variables set USE_CONGRESS_API=true
railway variables set USE_HOUSE_SCRAPER=true
railway variables set USE_MOCK_DATA=false
railway variables set ENABLE_EMAIL_ALERTS=false
railway variables set ENABLE_MONITORING=true

# Add PostgreSQL if not exists
echo "🗄️  Adding PostgreSQL database..."
if ! railway services | grep -q postgresql; then
    railway add postgresql
    echo "✅ PostgreSQL database added"
else
    echo "✅ PostgreSQL database already exists"
fi

# Deploy application
echo "🚀 Deploying application..."
railway up --detach

# Wait for deployment
echo "⏳ Waiting for deployment to complete..."
sleep 30

# Get deployment URL
echo "🌐 Getting deployment URL..."
RAILWAY_URL=$(railway domain 2>/dev/null || echo "URL not ready yet")

echo ""
echo "🎉 Deployment Complete!"
echo "======================================"
echo "📊 Project Status:"
railway status

echo ""
echo "🌐 Application URL: $RAILWAY_URL"
echo "📋 View logs: railway logs"
echo "🔧 Manage variables: railway variables"
echo "🗄️  Connect to DB: railway connect postgresql"

echo ""
echo "🧪 Test your deployment:"
echo "curl $RAILWAY_URL/health"
echo "curl $RAILWAY_URL/api/v1/info"

echo ""
echo "✅ Congressional Trading Intelligence System is now live!"