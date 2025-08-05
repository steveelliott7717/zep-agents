# deployment.md
"""
# Deployment Instructions for Fly.io

## 1. Install Fly CLI
```bash
curl -L https://fly.io/install.sh | sh
```

## 2. Login to Fly
```bash
fly auth login
```

## 3. Create your app
```bash
fly apps create your-agent-system
```

## 4. Set secrets (API keys)
```bash
fly secrets set ZEP_API_KEY="your_zep_api_key"
fly secrets set OPENAI_API_KEY="your_openai_api_key"
```

## 5. Deploy
```bash
fly deploy
```

## 6. Scale as needed
```bash
# Add more instances
fly scale count 2

# Add more memory/CPU
fly scale vm shared-cpu-2x
```

## 7. Monitor logs
```bash
fly logs
```

## 8. Set up custom domain (optional)
```bash
fly certs add yourdomain.com
```

## Webhook Configuration

After deployment, configure webhooks:

1. **Supabase Webhooks**: 
   - URL: `https://your-agent-system.fly.dev/webhooks/supabase`
   
2. **Scheduled Jobs** (using Fly.io scheduled machines or external cron):
   - URL: `https://your-agent-system.fly.dev/webhooks/schedule/{agent_id}`

## Environment Variables

You can also set additional environment variables:
```bash
fly secrets set ENVIRONMENT="production"
fly secrets set LOG_LEVEL="info"
```

## Database Persistence

For persistent storage beyond Zep graphs, consider:
1. Fly Postgres: `fly postgres create`
2. External database services
3. Fly Volumes for file storage

## Monitoring

1. Built-in metrics: `fly dashboard`
2. Custom monitoring endpoints in your app
3. Integration with services like Datadog or New Relic
"""