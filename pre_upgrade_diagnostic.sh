#!/bin/bash
echo "═══════════════════════════════════════════════════════════════════"
echo "🔍 MedIllustrator-AI - COMPLETE PRE-v3.1 UPGRADE DIAGNOSTIC"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

# ============================================================================
# 1. CURRENT ENVIRONMENT
# ============================================================================
echo "📍 1. CURRENT ENVIRONMENT"
echo "────────────────────────────────────────────────────────────────────"
echo "Active Project: $(gcloud config get-value project)"
echo "Region: $(gcloud config get-value compute/region)"
echo "Zone: $(gcloud config get-value compute/zone)"
echo "User: $(gcloud config get-value account)"
echo "Cloud Shell Directory: $(pwd)"
echo ""

# ============================================================================
# 2. EXISTING FILES IN CLOUD SHELL
# ============================================================================
echo "📁 2. EXISTING FILES IN CLOUD SHELL"
echo "────────────────────────────────────────────────────────────────────"
echo "Home Directory Contents:"
ls -lah ~/ | grep -E "(medill|medical|app|\.zip|\.py)" || echo "No relevant files found"
echo ""

echo "Checking common directories:"
for dir in ~/medillustrator* ~/medical* ~/app* ~/code*; do
  if [ -d "$dir" ]; then
    echo ""
    echo "✅ Found: $dir"
    ls -lh "$dir" | head -15
  fi
done

# Check for uploaded zips
echo ""
echo "Uploaded packages (*.zip):"
find ~/ -maxdepth 2 -name "*.zip" -type f 2>/dev/null || echo "No zip files found"
echo ""

# ============================================================================
# 3. CURRENT RUNNING SERVICE DETAILS
# ============================================================================
echo "🚀 3. CURRENT RUNNING SERVICE"
echo "────────────────────────────────────────────────────────────────────"

# Make sure we're in the right project
gcloud config set project medical-image-ai >/dev/null 2>&1

SERVICE_NAME="medillustrator"
REGION="europe-west1"

if gcloud run services describe $SERVICE_NAME --region $REGION >/dev/null 2>&1; then
    echo "Service Name: $SERVICE_NAME"
    echo "Region: $REGION"
    echo ""
    
    echo "Current Configuration:"
    echo "  URL: $(gcloud run services describe $SERVICE_NAME --region $REGION --format='value(status.url)')"
    echo "  Memory: $(gcloud run services describe $SERVICE_NAME --region $REGION --format='value(spec.template.spec.containers[0].resources.limits.memory)')"
    echo "  CPU: $(gcloud run services describe $SERVICE_NAME --region $REGION --format='value(spec.template.spec.containers[0].resources.limits.cpu)')"
    echo ""
    
    echo "Environment Variables (current):"
    gcloud run services describe $SERVICE_NAME --region $REGION \
      --format="value(spec.template.spec.containers[0].env)" | \
      grep -E "(VERSION|ART|CLIP|LANGCHAIN)" || echo "  Basic configuration"
    echo ""
    
    echo "Container Image:"
    gcloud run services describe $SERVICE_NAME --region $REGION \
      --format="value(spec.template.spec.containers[0].image)"
    echo ""
    
    echo "Last Deployment:"
    gcloud run services describe $SERVICE_NAME --region $REGION \
      --format="value(metadata.creationTimestamp)"
else
    echo "⚠️  Service not found or not accessible"
fi
echo ""

# ============================================================================
# 4. SECRETS CONFIGURATION
# ============================================================================
echo "🔐 4. SECRETS IN SECRET MANAGER"
echo "────────────────────────────────────────────────────────────────────"
gcloud secrets list --format="table(name,createTime)" 2>/dev/null || echo "No secrets found"
echo ""

# ============================================================================
# 5. STORAGE BUCKETS
# ============================================================================
echo "🗄️  5. STORAGE BUCKETS"
echo "────────────────────────────────────────────────────────────────────"
gsutil ls 2>/dev/null | grep -E "medical|medill" || echo "No relevant buckets found"
echo ""

# ============================================================================
# 6. ARTIFACT REGISTRY
# ============================================================================
echo "🐳 6. CONTAINER REGISTRY"
echo "────────────────────────────────────────────────────────────────────"
gcloud artifacts repositories list --location=europe-west1 \
  --format="table(name,format,createTime)" 2>/dev/null || echo "No repositories found"
echo ""

# Check for existing images
echo "Existing Container Images:"
gcloud artifacts docker images list europe-west1-docker.pkg.dev/medical-image-ai/medillustrator \
  --format="table(image,createTime,updateTime)" 2>/dev/null | head -10 || echo "No images found"
echo ""

# ============================================================================
# 7. CURRENT vs TARGET v3.1 STRUCTURE
# ============================================================================
echo "📊 7. CURRENT vs TARGET v3.1 STRUCTURE"
echo "────────────────────────────────────────────────────────────────────"

# Check if we have any existing project directory
PROJECT_DIR=$(find ~/ -maxdepth 1 -type d -name "*medill*" 2>/dev/null | head -1)

if [ -n "$PROJECT_DIR" ]; then
    echo "Existing Project Directory: $PROJECT_DIR"
    echo ""
    echo "Current Structure:"
    tree -L 2 "$PROJECT_DIR" 2>/dev/null || find "$PROJECT_DIR" -maxdepth 2 -type d | head -20
    echo ""
else
    echo "⚠️  No existing project directory found"
    echo "Will create fresh structure"
fi

echo ""
echo "Target v3.1 Structure (what we need to create):"
cat << 'TARGET_STRUCTURE'
medillustrator-v3.1/
├── app_v3_langgraph.py          [FROM UPLOAD]
├── requirements-enhanced.txt    [FROM UPLOAD]
├── Dockerfile                   [CREATE NEW]
├── .dockerignore                [CREATE NEW]
├── README_v31.md                [CREATE NEW]
│
├── config/                      [FROM UPLOAD + ENHANCE]
│   ├── settings.py             [EXISTING]
│   └── art_settings.py         [CREATE NEW - v3.1]
│
├── agents/                      [FROM UPLOAD]
├── workflows/                   [FROM UPLOAD]
├── core/                        [FROM UPLOAD]
├── data/                        [FROM UPLOAD]
│   └── ontology_terms.csv
│
├── training/                    [CREATE NEW - v3.1]
│   └── __init__.py
├── models/                      [CREATE NEW - v3.1]
│   ├── baseline/
│   ├── checkpoints/
│   └── production/
├── evaluation/                  [CREATE NEW - v3.1]
├── notebooks/                   [CREATE NEW - v3.1]
└── examples/                    [CREATE NEW - v3.1]
TARGET_STRUCTURE
echo ""

# ============================================================================
# 8. DEPENDENCY CHECK
# ============================================================================
echo "🔧 8. REQUIRED TOOLS & DEPENDENCIES"
echo "────────────────────────────────────────────────────────────────────"
echo "Python Version: $(python3 --version 2>/dev/null || echo 'Not found')"
echo "gcloud Version: $(gcloud --version | head -1)"
echo "Docker Available: $(command -v docker >/dev/null && echo 'Yes' || echo 'No (not needed)')"
echo "unzip Available: $(command -v unzip >/dev/null && echo 'Yes ✅' || echo 'No ❌')"
echo "tree Available: $(command -v tree >/dev/null && echo 'Yes' || echo 'No (optional)')"
echo ""

# ============================================================================
# 9. NETWORK & ACCESS CHECK
# ============================================================================
echo "🌐 9. NETWORK & ACCESS VERIFICATION"
echo "────────────────────────────────────────────────────────────────────"

# Test service endpoint
if [ -n "$SERVICE_NAME" ]; then
    SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region $REGION --format='value(status.url)' 2>/dev/null)
    if [ -n "$SERVICE_URL" ]; then
        echo "Testing service endpoint..."
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$SERVICE_URL" -m 10)
        if [ "$HTTP_CODE" = "200" ]; then
            echo "✅ Service is responding (HTTP $HTTP_CODE)"
        else
            echo "⚠️  Service returned HTTP $HTTP_CODE"
        fi
    fi
fi

# Test API connectivity
echo ""
echo "API Access Check:"
gcloud secrets list --limit 1 >/dev/null 2>&1 && echo "✅ Secret Manager: Accessible" || echo "⚠️  Secret Manager: Issue"
gsutil ls >/dev/null 2>&1 && echo "✅ Cloud Storage: Accessible" || echo "⚠️  Cloud Storage: Issue"
gcloud run services list --limit 1 >/dev/null 2>&1 && echo "✅ Cloud Run: Accessible" || echo "⚠️  Cloud Run: Issue"
echo ""

# ============================================================================
# 10. DISK SPACE & RESOURCES
# ============================================================================
echo "💾 10. AVAILABLE RESOURCES"
echo "────────────────────────────────────────────────────────────────────"
echo "Cloud Shell Disk Space:"
df -h ~ | grep -E "(Filesystem|/$|/home)" | head -2
echo ""
echo "Available Memory:"
free -h | grep -E "(total|Mem:)"
echo ""

# ============================================================================
# 11. RECENT ACTIVITY LOG
# ============================================================================
echo "📝 11. RECENT SERVICE LOGS (Last 5)"
echo "────────────────────────────────────────────────────────────────────"
if [ -n "$SERVICE_NAME" ]; then
    gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=$SERVICE_NAME" \
      --limit 5 \
      --format="table(timestamp.date('%Y-%m-%d %H:%M:%S'),severity,textPayload)" 2>/dev/null || \
      echo "No recent logs available"
fi
echo ""

# ============================================================================
# 12. UPGRADE READINESS CHECKLIST
# ============================================================================
echo "✅ 12. UPGRADE READINESS CHECKLIST"
echo "────────────────────────────────────────────────────────────────────"

READY_COUNT=0
TOTAL_CHECKS=8

# Check 1: Project access
if gcloud config get-value project >/dev/null 2>&1; then
    echo "✅ [1/8] Project access configured"
    ((READY_COUNT++))
else
    echo "❌ [1/8] Project access issue"
fi

# Check 2: Service running
if gcloud run services describe medillustrator --region europe-west1 >/dev/null 2>&1; then
    echo "✅ [2/8] Current service accessible"
    ((READY_COUNT++))
else
    echo "❌ [2/8] Service not accessible"
fi

# Check 3: Secrets configured
SECRET_COUNT=$(gcloud secrets list --format="value(name)" 2>/dev/null | wc -l)
if [ "$SECRET_COUNT" -gt 0 ]; then
    echo "✅ [3/8] Secrets configured ($SECRET_COUNT secrets)"
    ((READY_COUNT++))
else
    echo "❌ [3/8] No secrets found"
fi

# Check 4: Billing enabled
if gcloud billing projects describe $(gcloud config get-value project) --format="value(billingEnabled)" 2>/dev/null | grep -q "True"; then
    echo "✅ [4/8] Billing enabled"
    ((READY_COUNT++))
else
    echo "⚠️  [4/8] Billing status unclear"
fi

# Check 5: Storage available
if gsutil ls >/dev/null 2>&1; then
    echo "✅ [5/8] Cloud Storage accessible"
    ((READY_COUNT++))
else
    echo "❌ [5/8] Cloud Storage issue"
fi

# Check 6: unzip available
if command -v unzip >/dev/null 2>&1; then
    echo "✅ [6/8] unzip tool available"
    ((READY_COUNT++))
else
    echo "❌ [6/8] unzip not available (install: sudo apt-get install unzip)"
fi

# Check 7: Disk space
AVAILABLE_GB=$(df -BG ~ | awk 'NR==2 {print $4}' | sed 's/G//')
if [ "$AVAILABLE_GB" -gt 5 ]; then
    echo "✅ [7/8] Sufficient disk space (${AVAILABLE_GB}GB available)"
    ((READY_COUNT++))
else
    echo "⚠️  [7/8] Low disk space (${AVAILABLE_GB}GB available)"
fi

# Check 8: Network connectivity
if curl -s -o /dev/null -w "%{http_code}" https://www.google.com -m 5 | grep -q "200"; then
    echo "✅ [8/8] Network connectivity OK"
    ((READY_COUNT++))
else
    echo "❌ [8/8] Network connectivity issue"
fi

echo ""
echo "────────────────────────────────────────────────────────────────────"
echo "READINESS SCORE: $READY_COUNT/$TOTAL_CHECKS checks passed"

if [ $READY_COUNT -ge 6 ]; then
    echo "✅ SYSTEM READY για v3.1 upgrade!"
else
    echo "⚠️  Some issues detected - review above"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "📊 DIAGNOSTIC COMPLETE - Ready για STEP 2"
echo "═══════════════════════════════════════════════════════════════════"
