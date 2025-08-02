#!/bin/bash

# Congressional Trading Intelligence System - Railway Deployment Script
# Run this script in your terminal to deploy to Railway

echo "🚂 Congressional Trading Intelligence System - Railway Deployment"
echo "================================================================="
echo ""

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not found. Installing..."
    npm install -g @railway/cli
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install Railway CLI. Please install manually:"
        echo "   npm install -g @railway/cli"
        exit 1
    fi
fi

echo "✅ Railway CLI found: $(railway --version)"
echo ""

# Check login status
echo "🔐 Checking Railway login status..."
railway whoami > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "❌ Not logged in to Railway. Please run:"
    echo "   railway login"
    echo ""
    echo "Then run this script again."
    exit 1
fi

echo "✅ Logged in to Railway"
echo ""

# Show current directory and git status
echo "📁 Current directory: $(pwd)"
echo "📊 Git status:"
git status --porcelain
echo ""

# Check for required files
echo "🔍 Checking deployment files..."

required_files=("app.py" "Procfile" "railway.toml")
missing_files=()

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file found"
    else
        echo "❌ $file missing"
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -ne 0 ]; then
    echo ""
    echo "❌ Missing required files for deployment. Please ensure these files exist:"
    printf '   %s\n' "${missing_files[@]}"
    exit 1
fi

echo ""
echo "🎯 All deployment files ready!"
echo ""

# Ask user for confirmation
echo "🚀 Ready to deploy Congressional Trading Intelligence System to Railway?"
echo "   Project: handsome-cooperation"
echo "   Features: 531 Congressional Members, 1,755 Trades, Advanced Risk Analysis"
echo ""
read -p "Continue with deployment? (y/N): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Deployment cancelled."
    exit 1
fi

echo ""
echo "🚂 Starting Railway deployment..."
echo ""

# Link to existing project or create new one
echo "🔗 Linking to Railway project 'handsome-cooperation'..."
railway link

if [ $? -ne 0 ]; then
    echo "❌ Failed to link to Railway project. Try:"
    echo "   railway login"
    echo "   railway link"
    echo "   Or create a new project with: railway init"
    exit 1
fi

echo ""
echo "📦 Deploying to Railway..."
railway up

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 DEPLOYMENT SUCCESSFUL!"
    echo "================================================================="
    echo ""
    echo "🌐 Your Congressional Trading Intelligence System is now live!"
    echo ""
    echo "📊 API Endpoints:"
    echo "   • Dashboard: https://handsome-cooperation.railway.app/"
    echo "   • Members API: https://handsome-cooperation.railway.app/api/members"
    echo "   • Trades API: https://handsome-cooperation.railway.app/api/trades"
    echo "   • High-Risk API: https://handsome-cooperation.railway.app/api/high-risk"
    echo "   • Health Check: https://handsome-cooperation.railway.app/health"
    echo ""
    echo "🎯 Features Available:"
    echo "   ✅ 531 Congressional Members (435 House + 100 Senate)"
    echo "   ✅ 1,755 Trading Records ($750M+ volume)"
    echo "   ✅ 27 High-Risk Members identified"
    echo "   ✅ 84.6% STOCK Act compliance tracking"
    echo "   ✅ Real-time API access"
    echo "   ✅ Interactive web dashboard"
    echo ""
    echo "🔗 Railway Dashboard: https://railway.app/dashboard"
    echo ""
else
    echo ""
    echo "❌ DEPLOYMENT FAILED"
    echo "================================"
    echo ""
    echo "🔧 Troubleshooting steps:"
    echo "   1. Check Railway dashboard for build logs"
    echo "   2. Verify all files are committed to git"
    echo "   3. Check requirements.txt has correct dependencies"
    echo "   4. Ensure app.py is in the root directory"
    echo ""
    echo "📋 Common issues:"
    echo "   • Python version compatibility"
    echo "   • Missing dependencies in requirements.txt"
    echo "   • Port configuration issues"
    echo "   • Memory limits exceeded"
    echo ""
    echo "🆘 Get help:"
    echo "   • Railway docs: https://docs.railway.app"
    echo "   • Railway Discord: https://discord.gg/railway"
    echo ""
fi