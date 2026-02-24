#!/bin/bash

# Setup GitHub Repository Script
# This script helps you create a private GitHub repository and push your code

echo "=========================================="
echo "GitHub Repository Setup"
echo "=========================================="
echo ""

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "Error: git is not installed. Please install git first."
    exit 1
fi

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo "Warning: GitHub CLI (gh) is not installed."
    echo "You can install it from: https://cli.github.com/"
    echo ""
    echo "Or create the repository manually:"
    echo "1. Go to https://github.com/new"
    echo "2. Create a private repository"
    echo "3. Follow the instructions below"
    echo ""
    USE_GH=false
else
    USE_GH=true
fi

# Get repository name
read -p "Enter repository name (default: langchain-websocket-agentcore): " REPO_NAME
REPO_NAME=${REPO_NAME:-langchain-websocket-agentcore}

echo ""
echo "Repository name: $REPO_NAME"
echo ""

# Initialize git if not already initialized
if [ ! -d .git ]; then
    echo "Initializing git repository..."
    git init
    echo "✓ Git repository initialized"
else
    echo "✓ Git repository already initialized"
fi

# Add all files
echo ""
echo "Adding files to git..."
git add .

# Create initial commit
echo ""
echo "Creating initial commit..."
git commit -m "Initial commit: LangChain WebSocket agents on Bedrock AgentCore Runtime

- Basic WebSocket agent with LangChain
- Streaming WebSocket agent with token-by-token responses
- Strands framework integration
- Multiple authentication methods (SigV4, OAuth)
- Custom tools (timestamp, random, UUID, hashing, dates)
- Session management
- Comprehensive documentation"

echo "✓ Initial commit created"

# Create GitHub repository
if [ "$USE_GH" = true ]; then
    echo ""
    read -p "Create private GitHub repository now? (y/n): " CREATE_REPO
    
    if [ "$CREATE_REPO" = "y" ] || [ "$CREATE_REPO" = "Y" ]; then
        echo ""
        echo "Creating private GitHub repository..."
        gh repo create "$REPO_NAME" --private --source=. --remote=origin --push
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "=========================================="
            echo "✓ Repository created successfully!"
            echo "=========================================="
            echo ""
            echo "Repository URL: https://github.com/$(gh api user -q .login)/$REPO_NAME"
            echo ""
            echo "Next steps:"
            echo "1. Visit your repository on GitHub"
            echo "2. Add collaborators if needed"
            echo "3. Configure branch protection rules"
            echo ""
        else
            echo "Error creating repository. Please create it manually."
        fi
    else
        echo ""
        echo "Skipping repository creation."
        echo ""
        echo "To create the repository manually:"
        echo "1. Go to https://github.com/new"
        echo "2. Repository name: $REPO_NAME"
        echo "3. Make it Private"
        echo "4. Do NOT initialize with README, .gitignore, or license"
        echo "5. Click 'Create repository'"
        echo ""
        echo "Then run these commands:"
        echo "  git remote add origin https://github.com/YOUR_USERNAME/$REPO_NAME.git"
        echo "  git branch -M main"
        echo "  git push -u origin main"
    fi
else
    echo ""
    echo "=========================================="
    echo "Manual Setup Instructions"
    echo "=========================================="
    echo ""
    echo "1. Go to https://github.com/new"
    echo "2. Repository name: $REPO_NAME"
    echo "3. Description: LangChain WebSocket agents on Amazon Bedrock AgentCore Runtime"
    echo "4. Make it Private ✓"
    echo "5. Do NOT initialize with README, .gitignore, or license"
    echo "6. Click 'Create repository'"
    echo ""
    echo "Then run these commands:"
    echo "  git remote add origin https://github.com/YOUR_USERNAME/$REPO_NAME.git"
    echo "  git branch -M main"
    echo "  git push -u origin main"
    echo ""
fi

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
