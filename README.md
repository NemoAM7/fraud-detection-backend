# Fraud Detection API

A FastAPI-based API for real-time and batch fraud detection.

## Deployment to Vercel

### Prerequisites

1. [Vercel Account](https://vercel.com/signup)
2. [Vercel CLI](https://vercel.com/cli) (optional)

### Deployment Steps

#### Option 1: Using Vercel Dashboard

1. Push your code to a Git repository (GitHub, GitLab, or Bitbucket)
2. Log in to your Vercel account
3. Click on "New Project"
4. Import your Git repository
5. Configure the project:
   - Framework Preset: Other
   - Build Command: None
   - Output Directory: None
6. Add any required environment variables
7. Click "Deploy"

#### Option 2: Using Vercel CLI

1. Install Vercel CLI:
   ```
   npm i -g vercel
   ```

2. Log in to Vercel:
   ```
   vercel login
   ```

3. Deploy the project:
   ```
   vercel
   ```

4. Follow the CLI prompts to configure your deployment.

### Important Notes

1. **Database**: The current implementation uses SQLite which isn't suitable for production on Vercel's serverless environment. For production, consider using:
   - PostgreSQL on [Vercel Postgres](https://vercel.com/docs/storage/vercel-postgres)
   - [Supabase](https://supabase.com/)
   - [PlanetScale](https://planetscale.com/)
   - Other cloud database services

2. **ML Models**: The large ML model files (72MB) might be too large for Vercel. Consider:
   - Using a simplified model for the Vercel deployment
   - Hosting models on cloud storage (AWS S3, Google Cloud Storage)
   - Using a separate ML service API

3. **Environment Variables**: Set these in the Vercel dashboard:
   - `FRAUD_MODEL_TYPE`: Model type to use (undersampling, oversampling, smote, or mock)
   - `DATABASE_URL`: If using an external database

## Local Development

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the server:
   ```
   uvicorn server:app --reload
   ```

3. Access the API at http://localhost:8000 and the OpenAPI docs at http://localhost:8000/docs 