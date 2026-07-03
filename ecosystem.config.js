const path = require('path');

module.exports = {
  apps: [
    {
      name: 'embedding-service',
      script: 'uvicorn',
      args: 'app.main:app --host 0.0.0.0 --port 8800',
      interpreter: 'none',
      cwd: path.resolve(__dirname),
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: '2G',
      env: {
        NODE_ENV: 'production',
        PYTHONPATH: path.resolve(__dirname)
      },
      error_file: path.resolve(__dirname, './logs/pm2-error.log'),
      out_file: path.resolve(__dirname, './logs/pm2-out.log'),
      log_file: path.resolve(__dirname, './logs/pm2-combined.log'),
      time: true,
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
      merge_logs: true,
      restart_delay: 4000
    }
  ]
};
