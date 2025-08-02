# 🚀 Congressional Trading Intelligence System - Deployment Guide

## 🎯 Recommended Deployment: Railway

### **Why Railway?**
- ✅ **Free $5/month credit** (enough for development)
- ✅ **PostgreSQL included** automatically  
- ✅ **One-click deployment** from GitHub
- ✅ **Environment variables** built-in
- ✅ **Custom domain** provided
- ✅ **Auto-scaling** and monitoring

### **Step-by-Step Railway Deployment**

#### 1. Push to GitHub
```bash
# Initialize git if not already done
git init
git add .
git commit -m "Ready for Railway deployment"

# Push to GitHub
git remote add origin https://github.com/yourusername/congressional-trading-system.git
git push -u origin main
```

#### 2. Deploy to Railway
1. Go to **https://railway.app/**
2. Sign up with GitHub
3. Click **"New Project"**
4. Select **"Deploy from GitHub repo"**
5. Choose your congressional-trading-system repo
6. Railway automatically detects Python and deploys!

#### 3. Add Environment Variables
In Railway dashboard, go to **Variables** tab and add:

```bash
# API Keys
CONGRESS_GOV_API_KEY=GnEJVyPiswjccfl3Y9KHhwRVEeWDUnxOAVC4aMhD
FINNHUB_API_KEY=d26nr7hr01qvrairb710d26nr7hr01qvrairb71g

# Security (generate new ones for production)
SECRET_KEY=your_production_secret_key_here
JWT_SECRET_KEY=your_production_jwt_key_here

# Database (Railway provides automatically)
DATABASE_URL=${{Postgres.DATABASE_URL}}

# Application Settings
FLASK_ENV=production
PORT=5000
```

#### 4. Add PostgreSQL Database
1. In Railway project, click **"+ New"**
2. Select **"Database" → "PostgreSQL"**
3. Railway automatically connects it to your app

#### 5. Deploy!
- Railway automatically builds and deploys
- Get your live URL: `https://your-app-name.railway.app`
- View logs in Railway dashboard

---

## 🔄 Alternative: Render Deployment

### **Step-by-Step Render Deployment**

#### 1. Go to Render.com
1. Sign up at **https://render.com/**
2. Connect your GitHub account
3. Click **"New +" → "Web Service"**

#### 2. Configure Service
- **Repository**: Select your congressional-trading-system repo
- **Branch**: main
- **Runtime**: Python 3
- **Build Command**: `pip install -r requirements-production.txt`
- **Start Command**: `python src/api/app.py`

#### 3. Add Environment Variables
```bash
CONGRESS_GOV_API_KEY=GnEJVyPiswjccfl3Y9KHhwRVEeWDUnxOAVC4aMhD
FINNHUB_API_KEY=d26nr7hr01qvrairb710d26nr7hr01qvrairb71g
SECRET_KEY=your_production_secret_key
JWT_SECRET_KEY=your_production_jwt_key
FLASK_ENV=production
PORT=10000
```

#### 4. Add PostgreSQL Database
1. Click **"New +" → "PostgreSQL"**
2. Copy the database URL
3. Add as `DATABASE_URL` environment variable

---

## 🐳 Docker Deployment (Advanced)

### **Dockerfile**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements-production.txt .
RUN pip install --no-cache-dir -r requirements-production.txt

# Copy application
COPY . .

# Create directories
RUN mkdir -p logs models/phase2 analysis/network

# Environment
ENV FLASK_APP=src/api/app.py
ENV FLASK_ENV=production
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

EXPOSE 5000
CMD ["python", "src/api/app.py"]
```

### **Docker Compose** (with PostgreSQL & Redis)
```yaml
version: '3.8'
services:
  web:
    build: .
    ports:
      - "5000:5000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/congressional_trading
      - REDIS_URL=redis://redis:6379/0
      - CONGRESS_GOV_API_KEY=GnEJVyPiswjccfl3Y9KHhwRVEeWDUnxOAVC4aMhD
      - FINNHUB_API_KEY=d26nr7hr01qvrairb710d26nr7hr01qvrairb71g
    depends_on:
      - db
      - redis

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: congressional_trading
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
```

---

## 📊 Deployment Comparison

| Platform | Cost | Ease | Database | Performance | Recommendation |
|----------|------|------|----------|-------------|----------------|
| **Railway** | Free $5/mo | ⭐⭐⭐⭐⭐ | ✅ Included | ⭐⭐⭐⭐ | **Best for beginners** |
| **Render** | Free tier | ⭐⭐⭐⭐ | ✅ Free 90 days | ⭐⭐⭐⭐ | **Great alternative** |
| **Heroku** | $7/mo+ | ⭐⭐⭐ | ✅ Add-ons | ⭐⭐⭐⭐ | Classic choice |
| **DigitalOcean** | $5/mo+ | ⭐⭐⭐ | ✅ Managed | ⭐⭐⭐⭐⭐ | Production ready |
| **Docker** | Variable | ⭐⭐ | Manual setup | ⭐⭐⭐⭐⭐ | Full control |

---

## 🎯 **Recommended Approach**

### **For Quick Demo/Testing: Railway**
1. Push to GitHub (2 minutes)
2. Connect Railway (1 minute)  
3. Add environment variables (2 minutes)
4. **Live URL in 5 minutes!**

### **For Production: DigitalOcean App Platform**
- More control and performance
- Professional monitoring
- Custom domains and SSL
- Scalable infrastructure

---

## 🚀 **Post-Deployment Steps**

### 1. **Test Your Live Application**
```bash
# Test API endpoints
curl https://your-app.railway.app/health
curl https://your-app.railway.app/api/v1/info
```

### 2. **Monitor Performance**
- Check Railway/Render dashboard
- Monitor response times
- Watch error logs

### 3. **Set Up Custom Domain** (Optional)
- Add your custom domain in platform settings
- Configure DNS records
- Enable SSL certificate

### 4. **Enable Monitoring**
- Set up alerts for downtime
- Monitor database usage
- Track API response times

---

## 🎉 **Result**

After deployment, you'll have:

✅ **Live Congressional Trading Intelligence System**  
✅ **Real-time congressional data analysis**  
✅ **Interactive web dashboard**  
✅ **Secure API endpoints**  
✅ **Professional production environment**  
✅ **Scalable infrastructure**

**Your system will be accessible worldwide with a professional URL!**