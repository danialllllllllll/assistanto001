
#!/bin/bash
# Setup Replit Deployment for Permanent Public URL

echo "Setting up permanent public deployment..."

# Create deployment configuration
cat > .replit.deployment << 'EOF'
[deployment]
run = ["python", "train_advanced_ai.py"]
deploymentTarget = "cloudrun"

[[ports]]
localPort = 5000
externalPort = 80
EOF

echo "✓ Deployment configuration created"
echo ""
echo "To deploy with a permanent URL:"
echo "1. Click 'Deploy' button in Replit"
echo "2. Choose 'Autoscale' deployment"
echo "3. Your app will get a permanent *.replit.app URL"
echo "4. Optionally add a custom domain in deployment settings"
echo ""
echo "The deployment will be accessible from anywhere, including schools"
echo "(uses standard HTTPS port 443, bypasses most firewalls)"
