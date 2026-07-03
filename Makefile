.PHONY: help install start stop restart status logs monit uninstall clean

# Default target
help:
	@echo "Embedding Service Management"
	@echo ""
	@echo "Available targets:"
	@echo "  install     - Install dependencies and setup environment"
	@echo "  start       - Start the service with PM2"
	@echo "  stop        - Stop the service"
	@echo "  restart     - Restart the service"
	@echo "  status      - Show service status"
	@echo "  logs        - Show service logs"
	@echo "  monit       - Monitor service in real-time"
	@echo "  uninstall   - Remove PM2 process"
	@echo "  clean       - Clean logs and cache"
	@echo ""

# Install dependencies
install:
	@echo "Installing Python dependencies..."
	pip install -r requirements.txt
	@mkdir -p logs
	@echo "Installation complete!"

# Start service
start:
	@echo "Starting embedding service..."
	pm2 start ecosystem.config.js
	@echo "Service started!"
	@make status

# Stop service
stop:
	@echo "Stopping embedding service..."
	pm2 stop embedding-service
	@echo "Service stopped!"

# Restart service
restart:
	@echo "Restarting embedding service..."
	pm2 restart embedding-service
	@echo "Service restarted!"
	@make status

# Show status
status:
	@echo "Service status:"
	@pm2 list | grep -E "embedding-service|App name"

# Show logs
logs:
	@pm2 logs embedding-service

# Monitor service
monit:
	@pm2 monit

# Uninstall/remove from PM2
uninstall:
	@echo "Removing embedding service from PM2..."
	pm2 delete embedding-service
	@echo "Service removed!"

# Clean logs and cache
clean:
	@echo "Cleaning logs..."
	rm -rf logs/*.log
	@mkdir -p logs
	@echo "Logs cleaned!"
