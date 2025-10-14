#!/bin/bash

# ============================================================================
# MedIllustrator-AI v3.1 - FIXED COMPLETE STRUCTURE ANALYSIS
# ============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Header
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}📊 MedIllustrator-AI v3.1 - COMPLETE STRUCTURE ANALYSIS (FIXED)${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════${NC}"
echo ""

# ============================================================================
# 1. TREE STRUCTURE
# ============================================================================
echo -e "${BLUE}🌳 1. PROJECT TREE STRUCTURE${NC}"
echo "────────────────────────────────────────────────────────────────────"

if command -v tree &> /dev/null; then
    tree -L 3 -h --du --dirsfirst
else
    echo "Tree not installed, using find..."
    find . -maxdepth 3 -type d | sort
fi
echo ""

# ============================================================================
# 2. ROOT LEVEL FILES
# ============================================================================
echo -e "${BLUE}📄 2. ROOT LEVEL FILES (Detailed)${NC}"
echo "────────────────────────────────────────────────────────────────────"
ls -lh --time-style="+%b %d %H:%M" *.py *.txt *.sh Dockerfile .dockerignore .env 2>/dev/null | \
    awk '{printf "%-50s %5s  %s %s %s\n", $9, $5, $6, $7, $8}' || echo "Some files not found"
echo ""

# ============================================================================
# 3. ALL DIRECTORIES
# ============================================================================
echo -e "${BLUE}📁 3. ALL DIRECTORIES (with sizes)${NC}"
echo "────────────────────────────────────────────────────────────────────"
du -sh */ 2>/dev/null | sort -h | awk '{printf "%-40s %s\n", $2, $1}'
echo ""

# ============================================================================
# 4. KEY FILES VERIFICATION (FIXED)
# ============================================================================
echo -e "${BLUE}🔑 4. KEY FILES VERIFICATION${NC}"
echo "────────────────────────────────────────────────────────────────────"

check_file() {
    local file=$1
    local description=$2
    
    if [[ -f "$file" ]]; then
        local size=$(ls -lh "$file" | awk '{print $5}')
        local date=$(ls -l --time-style="+%b %d %H:%M" "$file" | awk '{print $6, $7, $8}')
        echo -e "${GREEN}✅${NC} $description"
        echo "   File: $file"
        echo "   Size: $size, Modified: $date"
    else
        echo -e "${RED}❌${NC} $description"
        echo "   File: $file - ${RED}NOT FOUND${NC}"
    fi
    echo ""
}

# Check all key files with CORRECT filenames
check_file "app_v3_langgraph.py" "Main Application"
check_file "requirements.txt" "Python Requirements"
check_file "Dockerfile" "Docker Configuration"
check_file ".dockerignore" "Docker Ignore File"
check_file ".env" "Environment Variables"

check_file "config/settings.py" "Main Settings"
check_file "config/art_settings.py" "ART Settings"

check_file "agents/medical_terms_agent.py" "Medical Terms Agent"
check_file "agents/bloom_agent.py" "Bloom's Taxonomy Agent"
check_file "agents/cognitive_load_agent.py" "Cognitive Load Agent"

check_file "data/ontology_terms.csv" "Medical Ontology CSV"

# FIXED: Check for correct filename
check_file "workflows/med_assessment_graph.py" "Medical Assessment Graph (LangGraph)"
check_file "workflows/state_definitions.py" "State Definitions"
check_file "workflows/node_implementations.py" "Node Implementations"

check_file "core/enhanced_visual_analysis.py" "Enhanced Visual Analysis"
check_file "core/medical_ontology.py" "Medical Ontology Core"

check_file "utils/ontology_loader.py" "Ontology Loader Utility"

# ============================================================================
# 5. PYTHON FILES ANALYSIS
# ============================================================================
echo -e "${BLUE}🐍 5. PYTHON FILES DETAILED${NC}"
echo "────────────────────────────────────────────────────────────────────"

echo "Python files found:"
find . -name "*.py" -type f ! -path "./*/__pycache__/*" ! -name "*backup*" ! -name "*old*" | \
    while read file; do
        size=$(ls -lh "$file" | awk '{print $5}')
        date=$(ls -l --time-style="+%b %d" "$file" | awk '{print $6, $7}')
        printf "%-60s %6s  %s\n" "$file" "$size" "$date"
    done

echo ""
echo "Line counts (main files):"
total_lines=0
for file in $(find . -name "*.py" -type f ! -path "./*/__pycache__/*" ! -name "*backup*" ! -name "*old*" | sort); do
    if [[ -f "$file" ]]; then
        lines=$(wc -l < "$file")
        total_lines=$((total_lines + lines))
        printf "%6d lines  %s\n" "$lines" "$file"
    fi
done | sort -rn | head -15

echo ""
echo "Total Python lines: $total_lines"
echo ""

# ============================================================================
# 6. DEPLOYMENT READINESS CHECK (FIXED)
# ============================================================================
echo -e "${BLUE}✅ 6. DEPLOYMENT READINESS CHECK${NC}"
echo "────────────────────────────────────────────────────────────────────"

passed=0
total=0

check_exists() {
    local file=$1
    local name=$2
    total=$((total + 1))
    
    if [[ -e "$file" ]]; then
        echo -e "${GREEN}✅${NC} $name exists"
        passed=$((passed + 1))
    else
        echo -e "${RED}❌${NC} $name missing"
    fi
}

# FIXED: Proper checks with correct filenames
check_exists "app_v3_langgraph.py" "Main app file"
check_exists "requirements.txt" "Requirements file"
check_exists "Dockerfile" "Dockerfile"
check_exists "config" "Config directory"
check_exists "agents" "Agents directory"
check_exists "data" "Data directory"
check_exists "data/ontology_terms.csv" "Ontology CSV"
check_exists "agents/medical_terms_agent.py" "Medical terms agent"
check_exists "workflows" "Workflows directory"
check_exists "workflows/med_assessment_graph.py" "Assessment graph (CORRECTED)"
check_exists "core" "Core directory"
check_exists "core/enhanced_visual_analysis.py" "Visual analysis"

echo ""
echo -e "Readiness Score: ${GREEN}$passed/$total${NC}"

if [[ $passed -eq $total ]]; then
    echo -e "${GREEN}🎉 PROJECT IS FULLY READY FOR DEPLOYMENT!${NC}"
elif [[ $passed -ge $((total * 3 / 4)) ]]; then
    echo -e "${YELLOW}⚠️  PROJECT IS MOSTLY READY (some optional files missing)${NC}"
else
    echo -e "${RED}❌ PROJECT NEEDS ATTENTION (missing critical files)${NC}"
fi
echo ""

# ============================================================================
# 7. DISK USAGE SUMMARY
# ============================================================================
echo -e "${BLUE}💾 7. DISK USAGE SUMMARY${NC}"
echo "────────────────────────────────────────────────────────────────────"
echo "Current directory total size:"
du -sh . 2>/dev/null

echo ""
echo "Breakdown by category:"
python_size=$(find . -name "*.py" -type f ! -path "./*/__pycache__/*" -exec du -ch {} + 2>/dev/null | tail -1 | awk '{print $1}')
data_size=$(du -sh data 2>/dev/null | awk '{print $1}')
cache_size=$(du -sh cache __pycache__ */__pycache__ 2>/dev/null | awk '{sum+=$1} END {print sum"K"}')

echo "  Python source: $python_size"
echo "  Data files: $data_size"
echo "  Cache/temp: $cache_size"
echo ""

# ============================================================================
# 8. RECENT MODIFICATIONS
# ============================================================================
echo -e "${BLUE}🕐 8. RECENTLY MODIFIED FILES (Last 10)${NC}"
echo "────────────────────────────────────────────────────────────────────"
find . -type f ! -path "./*/__pycache__/*" ! -path "./.git/*" -printf '%T+ %p\n' 2>/dev/null | \
    sort -r | head -10 | \
    awk '{printf "%-24s  %s\n", $1, $2}'
echo ""

# ============================================================================
# 9. DOCKER & DEPLOYMENT FILES
# ============================================================================
echo -e "${BLUE}🐳 9. DOCKER & DEPLOYMENT FILES${NC}"
echo "────────────────────────────────────────────────────────────────────"

if [[ -f "Dockerfile" ]]; then
    echo -e "${GREEN}✅${NC} Dockerfile found"
    echo "   Size: $(ls -lh Dockerfile | awk '{print $5}')"
    echo "   Lines: $(wc -l < Dockerfile)"
    echo ""
    echo "   Dockerfile content preview:"
    head -5 Dockerfile | sed 's/^/   | /'
else
    echo -e "${RED}❌${NC} Dockerfile not found"
fi
echo ""

if [[ -f ".dockerignore" ]]; then
    echo -e "${GREEN}✅${NC} .dockerignore found"
else
    echo -e "${YELLOW}⚠️${NC}  .dockerignore not found (optional)"
fi
echo ""

# ============================================================================
# 10. SUMMARY
# ============================================================================
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}📊 SUMMARY${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════${NC}"

total_dirs=$(find . -type d | wc -l)
total_files=$(find . -type f ! -path "./*/__pycache__/*" ! -path "./.git/*" | wc -l)
total_size=$(du -sh . 2>/dev/null | awk '{print $1}')
python_files=$(find . -name "*.py" -type f ! -path "./*/__pycache__/*" | wc -l)

echo "Total Directories: $total_dirs"
echo "Total Files: $total_files"
echo "Total Size: $total_size"
echo "Python Files: $python_files"
echo "Total Lines of Code: $total_lines"
echo "Deployment Readiness: $passed/$total checks passed"
echo ""
echo "Project Location: $(pwd)"
echo "Analysis Date: $(date)"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════${NC}"

# Finish
