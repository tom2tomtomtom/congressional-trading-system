# Congressional Trading Intelligence System - Enhanced Configuration

## Project Context
**Product**: Congressional Trading Intelligence Platform
**Mission**: Transparent analysis of congressional trading patterns for research and accountability
**Focus**: Educational transparency, data accuracy, and democratic accountability

## Claude Suite Integration
This project uses the enhanced Claude Productivity Suite with three-layer context system:

### 📁 Context Layers
- **Standards**: `~/.claude-suite/standards/` (Global development standards)
- **Project**: `.claude-suite/project/` (Product-specific documentation)  
- **Specs**: `.claude-suite/project/specs/` (Individual feature specifications)

## Available Commands

### 🚀 Development Commands (Primary)
```bash
/daily-dev                              # Run daily development workflow
/analyze-codebase                       # Deep codebase analysis and quality check
/clean-codebase                         # Automated code formatting and cleanup
/pre-deploy-check                       # Comprehensive deployment validation
```

### 🔧 Intelligence System Commands
```bash
/test-intelligence-engine               # Run congressional analysis tests
/update-dashboard-data                  # Refresh dashboard with latest data
/validate-data-integrity                # Check congressional data accuracy
/run-correlation-analysis               # Committee-trading pattern analysis
```

### 🧠 Planning & Development Commands  
```bash
/plan-product                          # Strategic product planning
/create-spec                           # Detailed feature specifications
/analyze-dependencies                  # Package and API dependency review
/performance-benchmark                 # System performance analysis
```

### 📊 Research & Analysis Commands
```bash
/generate-research-report              # Academic-style analysis report
/export-data-analysis                  # Statistical analysis export
/compliance-audit                      # STOCK Act compliance review
/transparency-metrics                  # Public accountability metrics
```

## Project Architecture

### Current Implementation Status
- ✅ **6-Tab Dashboard** - Professional HTML/CSS/JS interface
- ✅ **15+ Congressional Members** - Expanded database with party/state/chamber data
- ✅ **Committee Tracking** - Leadership positions and oversight areas
- ✅ **Active Legislation** - 7 major bills with market impact analysis
- ✅ **Pattern Recognition** - Committee-trading correlation analysis
- ✅ **Intelligence Engine** - Python-based analysis with ML foundation

### Tech Stack Summary
- **Backend**: Python 3.13+ with pandas, scikit-learn, tensorflow
- **Frontend**: HTML/CSS/JS dashboard (React migration planned)
- **Data**: Sample data structure ready for real-time API integration
- **Analysis**: Congressional trading pattern recognition and scoring

## Key Principles
1. **Educational Focus**: All analysis for transparency and research purposes
2. **Data Accuracy**: Only publicly disclosed STOCK Act information
3. **Ethical Standards**: Promote accountability without harmful speculation
4. **Open Source**: Transparent methodology and reproducible research

## Development Workflow
- **Main Branch**: Stable version with core features
- **Feature Branches**: Active development (currently: `feature/data-expansion`)
- **Daily Workflow**: Use `/daily-dev` for routine development tasks
- **Quality Gates**: Pre-deployment validation with comprehensive checklists

## Quick References
- Mission: `.claude-suite/project/mission.md`
- Roadmap: `.claude-suite/project/roadmap.md`  
- Tech Stack: `.claude-suite/project/tech-stack.md`
- Workflows: `.claude-suite/workflows/daily-dev.md`
- Checklists: `.claude-suite/project/checklists/pre-deploy.md`

## Testing & Validation
```bash
python3 src/analysis/congressional_analysis.py    # Test analysis engine
cd src/dashboard && python3 -m http.server 8000   # Test dashboard
/pre-deploy-check                                 # Full validation suite
```

## Educational & Legal Framework
- All data from official STOCK Act disclosures and public records
- Educational disclaimers throughout platform
- No trading advice or financial recommendations
- Compliance with transparency and research regulations

## Success Metrics
- **Coverage**: 535+ congressional members (currently 15+)
- **Accuracy**: 95%+ data accuracy with real-time compliance
- **Performance**: <2 second dashboard response times
- **Research Impact**: Academic citations and investigative journalism
- **Public Engagement**: Educational transparency content
---

## 🚀 Ultimate Dev System Integration

This project is configured with the Ultimate Dev System for enhanced AI-assisted development.

### Tool Routing (python project)
- **Cursor**: UI development, quick changes (0-15K lines)
- **Augment**: Large refactoring, cross-file changes (15K+ lines)  
- **Claude Code**: Testing, automation, deployment

### Available Specialists
Use natural language to activate specialists:
- "Build the UI" → frontend-developer + ui-engineer
- "Fix database issues" → database-optimizer + debugger
- "Add security" → security-auditor + auth-engineer
- "Optimize performance" → performance-engineer + perf-optimizer

### Quick Commands
- `~/ultimate-dev-system/smart-workflow.sh` - Context-aware recommendations
- `~/ultimate-dev-system/quick-switch.sh congressional-trading-system` - Switch to this project
- Tell Claude: "Continue working on congressional-trading-system" for smart continuation

### Configuration Files
- `.cursor/rules/` - Cursor-specific rules and specialist routing
- `.claude/commands/` - Custom Claude commands for this project
- `.augment/config.json` - Augment configuration and preferences
